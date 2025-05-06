import json
import os
from PIL import Image
from torch.utils.data import Dataset
import random

class MyDataset2(Dataset):
    def __init__(self, transform=None):

        with open('/workspace/test/ProSrc/data/flicker/prompt.json', 'r') as f:
            self.data = json.load(f)
        
        self.image_paths = list(self.data.keys())
        self.prompts = [self.data[path][0] for path in self.image_paths]  

        self.transform = transform

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, idx):

        image_path = self.image_paths[idx]
        prompt = self.prompts[idx]

        if random.random() < 0.3:
            prompt = ""

        img = Image.open(image_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        img = img * 2 - 1

        return dict(jpg=img, txt=prompt, hint=img)

