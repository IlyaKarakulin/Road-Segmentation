import os

from PIL import Image
from pathlib import Path
import torch.utils.data as data
import torchvision.transforms.v2 as tfs_v2
import torch


class SegmentDataset(data.Dataset):
    def __init__(self, path, mode='train'):
        self.size = 1024
        self.path_img = Path(path) # Path(os.path.join(path, 'images'))
        self.path_mask = Path(path + '_labels')
        IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}
        
        images_dict = {p.stem: p for p in self.path_img.rglob('*') if p.suffix.lower() in IMAGE_EXTS}
        masks_dict = {p.stem: p for p in self.path_mask.rglob('*') if p.suffix.lower() in IMAGE_EXTS}

        all_keys = sorted(set(images_dict) & set(masks_dict))
        
        self.images = [images_dict[k] for k in all_keys]
        self.masks  = [masks_dict[k] for k in all_keys]

        # self.length = len(self.images)
        if mode=='train':
            self.length = 200
        elif mode=='test':
            self.length = len(self.images)
        elif mode=='val':
            self.length = len(self.images)
        

    def __getitem__(self, item):
        path_img, path_mask = self.images[item], self.masks[item]
        img = Image.open(path_img).convert('RGB')
        mask = Image.open(path_mask).convert('L')

        img_transformer = tfs_v2.Compose([
            tfs_v2.Resize((self.size, self.size), interpolation=tfs_v2.InterpolationMode.BILINEAR),
            tfs_v2.ToImage(),
            tfs_v2.ToDtype(torch.float32, scale=True)
        ])
        
        mask_transformer = tfs_v2.Compose([
            tfs_v2.Resize((self.size, self.size), interpolation=tfs_v2.InterpolationMode.NEAREST),
            tfs_v2.ToImage(),
            tfs_v2.ToDtype(torch.float32)
        ])

        transform_img = img_transformer(img)
        transform_mask = mask_transformer(mask)

        transform_mask[transform_mask < 250] = 1
        transform_mask[transform_mask >= 250] = 0

        return transform_img, transform_mask

    def __len__(self):
        return self.length