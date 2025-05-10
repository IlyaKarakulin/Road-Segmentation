import os

import gc
import torch

from model import Segmentator

gc.collect()
torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

path_to_train = './data/train'
path_to_val = './data/val'

model = Segmentator('cuda:3')

model.train(path_to_train=path_to_train,
            path_to_val=path_to_val,
            batch_size=4,
            lr=0.0005,
            num_epoch=2)