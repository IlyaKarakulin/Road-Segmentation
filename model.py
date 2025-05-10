import os

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt

from dataset import SegmentDataset


class UNetModel(nn.Module):
    class _TwoConvLayers(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
            )

        def forward(self, x):
            return self.model(x)

    class _EncoderBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.block = UNetModel._TwoConvLayers(in_channels, out_channels)
            self.max_pool = nn.MaxPool2d(2)

        def forward(self, x):
            x = self.block(x)
            y = self.max_pool(x)
            return y, x


    class _DecoderBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.transpose = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            self.block = UNetModel._TwoConvLayers(in_channels + out_channels, out_channels)

        def forward(self, x, skip_connection):
            x = self.transpose(x)
            # diffY = skip_connection.size()[2] - x.size()[2]
            # diffX = skip_connection.size()[3] - x.size()[3]
            # x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2,
            #                           diffY // 2, diffY - diffY // 2])
            
            u = torch.cat([x, skip_connection], dim=1)
            u = self.block(u)
            return u


    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.enc_block1 = self._EncoderBlock(in_channels, 64)
        self.enc_block2 = self._EncoderBlock(64, 128)
        self.enc_block3 = self._EncoderBlock(128, 256)
        # self.enc_block4 = self._EncoderBlock(256, 512)

        # Текущий bottleneck: вход 256 (с enc_block3), выход 512
        self.bottleneck = self._TwoConvLayers(256, 512)

        # self.dec_block1 = self._DecoderBlock(1024, 512)
        self.dec_block2 = self._DecoderBlock(512, 256)
        self.dec_block3 = self._DecoderBlock(256, 128)
        self.dec_block4 = self._DecoderBlock(128, 64)

        self.out = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x, y1 = self.enc_block1(x)
        x, y2 = self.enc_block2(x)
        x, y3 = self.enc_block3(x)
        # x, y4 = self.enc_block4(x)

        x = self.bottleneck(x)

        # x = self.dec_block1(x, y4)
        x = self.dec_block2(x, y3)
        x = self.dec_block3(x, y2)
        x = self.dec_block4(x, y1)

        return self.out(x)


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num = targets.size(0)
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2 * (intersection.sum(1) + self.smooth) / (m1.sum(1) + m2.sum(1) + self.smooth)
        score = 1 - score.sum() / num
        return score


class Segmentator():
    def __init__(self, device='cpu'):
        self.device = device
        self.model = UNetModel(in_channels=3, num_classes=1).to(self.device)
        # self.model = torch.compile(
        #     self.model,
        #     mode="max-autotune",
        #     fullgraph=True
        # )

        self.writer = None 

        if self.device == 'cpu':
            self.num_workers = 0
            self.pin_memory = False
        else:
            self.num_workers = 6
            self.pin_memory = True

    def train(self, path_to_train: str, path_to_val: str, num_epoch=100, batch_size=64, lr=0.001):
        log_dir = f'meta_data/metrics_lr{lr}_bs{batch_size}'
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

        segment_dataset = SegmentDataset(path_to_train, 'train')
        dataloader_train = data.DataLoader(segment_dataset, batch_size=batch_size, shuffle=True, 
                                           num_workers=self.num_workers, pin_memory=self.pin_memory)
        segment_val = SegmentDataset(path_to_val, 'val')
        dataloader_val = data.DataLoader(segment_val, batch_size=batch_size, shuffle=False,
                                          num_workers=self.num_workers, pin_memory=self.pin_memory)

        self.optimizer = optim.AdamW(params=self.model.parameters(), lr=lr, weight_decay=1e-4)
        loss_1 = nn.BCEWithLogitsLoss()
        loss_2 = SoftDiceLoss()

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epoch,
            eta_min=1e-6
        )

        self.model.train()
        best_val_iou = 0.0

        for count_epoch in range(num_epoch):
            train_metrics = self.run_epoch(count_epoch, dataloader_train, self.optimizer, loss_1, loss_2, is_train=True)
            val_metrics = self.run_epoch(count_epoch, dataloader_val, None, loss_1, loss_2, is_train=False)

            self.scheduler.step() #val_metrics['Val_Loss'])
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Learning Rate', current_lr, count_epoch)

            self.__log_metrics(train_metrics, val_metrics, count_epoch)
            self.__log_weights_and_grads(count_epoch)
                        
            if val_metrics["Val_IoU"] > best_val_iou:
                self.__save_model("best")
                
            print(
                f"Lr = {round(current_lr, 4)} | "
                f"Train Loss: {train_metrics['Train_Loss']:.4f} | "
                f"Train IoU: {train_metrics['Train_IoU']:.4f} | "
                f"Val Loss: {val_metrics['Val_Loss']:.4f} | "
                f"Val IoU: {val_metrics['Val_IoU']:.4f}"
            )

            self.__save_model("lost")

        self.writer.close()

    def run_epoch(self, epoch_num, dataloader, optimizer, criterion_bce, criterion_dice, is_train=True):
        if is_train:
            self.model.train()
            mode_str = "Train"
        else:
            self.model.eval()
            mode_str = "Val"

        epoch_loss = 0.0
        all_iou_scores = []
        
        progress_bar = tqdm(dataloader, desc=f"{epoch_num+1}", leave=is_train, ncols=100)

        for x_batch, y_batch in progress_bar:
            x_batch = x_batch.to(self.device, non_blocking=True if self.pin_memory else False)
            y_batch = y_batch.to(self.device, non_blocking=True if self.pin_memory else False)

            if is_train:
                optimizer.zero_grad()
            
            with torch.set_grad_enabled(is_train):
                predictions = self.model(x_batch)
                loss_bce = criterion_bce(predictions, y_batch)
                loss_dice = criterion_dice(predictions, y_batch)
                loss = loss_bce + loss_dice

            with torch.no_grad():
                probs = torch.sigmoid(predictions)
                preds_binary = (probs > 0.5).float()
                
                intersection = (preds_binary * y_batch).sum(dim=[1,2,3])
                union = preds_binary.sum(dim=[1,2,3]) + y_batch.sum(dim=[1,2,3]) - intersection
                
                smooth_iou = 1e-6
                iou_batch = (intersection + smooth_iou) / (union + smooth_iou)
                all_iou_scores.extend(iou_batch.cpu().tolist())

            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_loss += loss.item() * x_batch.size(0)
            
            if is_train:
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_epoch_loss = epoch_loss / len(dataloader.dataset)
        
        metrics = {f"{mode_str}_Loss": avg_epoch_loss}

        avg_iou = np.mean(all_iou_scores) if all_iou_scores else 0.0
        metrics[f"{mode_str}_IoU"] = avg_iou
        
        return metrics
    
    def __log_metrics(self, train_metrics, val_metrics, epoch):
        for metric, value in train_metrics.items():
            self.writer.add_scalar(f'Train/{metric.split("_")[1]}', value, epoch)
            self.writer.add_scalar(f'Train/{metric.split("_")[1]}', value, epoch)

        for metric, value in val_metrics.items():
            self.writer.add_scalar(f'Val/{metric.split("_")[1]}', value, epoch)
            self.writer.add_scalar(f'Val/{metric.split("_")[1]}', value, epoch)

    def __log_weights_and_grads(self, epoch):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f"Weights/{name}", param.clone().cpu().data.numpy(), epoch)
                if param.grad is not None:
                    self.writer.add_histogram(
                        f"Gradients/{name}", 
                        param.grad.clone().cpu().data.numpy(), 
                        epoch
                    )

    def __save_model(self, name):
        model_dir = './meta_data/models'
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{name}.pth")
        
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }

        torch.save(state, model_path)

    def load_model(self, path_to_checkpoint):
        if not os.path.exists(path_to_checkpoint):
            print(f"Checkpoint file not found: {path_to_checkpoint}")
            return

        checkpoint = torch.load(path_to_checkpoint, map_location=torch.device(self.device), weights_only=True)
        
        self.model = UNetModel(in_channels=3, num_classes=1).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        print(f"Model loaded from {path_to_checkpoint}")

    def test(self, path_to_test_data: str, batch_size=1, save_dir="test_results"):
        os.makedirs(save_dir, exist_ok=True)
        test_dataset = SegmentDataset(path_to_test_data, 'test')
        print(f"Test dataset size: {test_dataset.__len__()} samples")

        test_loader = data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )

        self.model.eval()
        all_ious = []
        with torch.no_grad():
            for batch_idx, (x_batch, y_batch) in enumerate(tqdm(test_loader, desc="Testing", ncols=100)):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                predictions = self.model(x_batch)
                probs = torch.sigmoid(predictions)
                preds_binary = (probs > 0.5).float()

                intersection = (preds_binary * y_batch).sum(dim=[1, 2, 3])
                union = preds_binary.sum(dim=[1, 2, 3]) + y_batch.sum(dim=[1, 2, 3]) - intersection
                iou = (intersection + 1e-6) / (union + 1e-6)
                all_ious.extend(iou.cpu().tolist())
        
                if batch_idx < 10:
                    self.visualize_batch(x_batch, y_batch, preds_binary, save_path=os.path.join(save_dir, f"batch_{batch_idx}.png"))

        mean_iou = np.mean(all_ious)
        print(f"Mean IoU on test: {mean_iou:.4f}")

    def visualize_batch(self, x_batch, y_batch, preds_binary, save_path=None, num_examples=4):
        batch_size = min(x_batch.size(0), num_examples)
        fig, axs = plt.subplots(batch_size, 3, figsize=(15, 3 * batch_size))
        if batch_size == 1:
            axs = [axs]
        for i in range(batch_size):
            img = x_batch[i].detach().cpu()
            img = TF.to_pil_image(img)
            gt = y_batch[i, 0].detach().cpu().numpy()
            pr = preds_binary[i, 0].detach().cpu().numpy()

            axs[i][0].imshow(img)
            axs[i][0].set_title('Input')
            axs[i][0].axis('off')

            axs[i][1].imshow(gt, cmap='gray')
            axs[i][1].set_title('Ground Truth')
            axs[i][1].axis('off')

            axs[i][2].imshow(pr, cmap='gray')
            axs[i][2].set_title('Prediction')
            axs[i][2].axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()
