import json
import cv2
import numpy as np
from PIL import Image
import torch
import random
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, transform=None):
        self.data = []
        with open('/workspace/test/ProSrc/data/LSDIR/prompt1.json', 'rt') as f:
            self.data = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        #source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']
        if random.random() < 0.3:
            prompt = ""
        img = Image.open(target_filename).convert("RGB")
        target = self.transform(img)
        target = target * 2 - 1

        return dict(jpg=target, txt=prompt, hint=target)
