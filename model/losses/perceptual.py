import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from .lpips import LPIPS
from pytorch_msssim import ms_ssim
# from taming.modules.discriminator.model import NLayerDiscriminator, weights_init

class LPIPS_loss(nn.Module):
    def __init__(self, metric="mse", perceptual_weight=1.0):
        super().__init__()
        if metric == "mse":
            self.metric = nn.MSELoss()
        elif metric == "ms-ssim":
            self.metric = ms_ssim
        else:
            raise NotImplementedError(f"{metric} is not implemented!")
        self.perceptual_loss = LPIPS().eval() 
        self.perceptual_weight = perceptual_weight

    def padding_and_trimming(self, x, x_rec):
        _, _, H, W = x.size()
        x_rec = F.pad(x_rec, (15, 15, 15, 15), mode='replicate')
        x = F.pad(x, (15, 15, 15, 15), mode='replicate')
        _, _, H_pad, W_pad = x.size()
        top = random.randrange(0, 16)
        bottom = H_pad - random.randrange(0, 16)
        left = random.randrange(0, 16)
        right = W_pad - random.randrange(0, 16)
        x_rec = F.interpolate(x_rec[:, :, top:bottom, left:right],
        size=(H, W), mode='bicubic', align_corners=False)
        x = F.interpolate(x[:, :, top:bottom, left:right],
        size=(H, W), mode='bicubic', align_corners=False)
        return  x, x_rec

    def forward(self, reconstructions, inputs):
        inputs, reconstructions = self.padding_and_trimming(inputs, reconstructions)
        p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())

        #  = torch.nn.functional.mse_loss(inputs.contiguous() , reconstructions.contiguous())
        if self.metric == ms_ssim:
            distortion = self.metric(reconstructions.contiguous(), inputs.contiguous(), data_range=2)
            rec_loss = 1 - distortion
        else:
            rec_loss = self.metric(reconstructions.contiguous(), inputs.contiguous())
            # distortion =  255**2 * out["mse_loss"]
        percptual_loss = rec_loss.mean() + self.perceptual_weight * p_loss.mean()
        
        return rec_loss, p_loss, percptual_loss

class LPIPS_loss_2(nn.Module):
    def __init__(self, metric="mse", perceptual_weight=1.0, perceptual_weight_d=1.0):
        super().__init__()
        if metric == "mse":
            self.metric = nn.MSELoss()
        elif metric == "ms-ssim":
            self.metric = ms_ssim
        else:
            raise NotImplementedError(f"{metric} is not implemented!")
        self.perceptual_loss = LPIPS().eval() 
        self.perceptual_weight = perceptual_weight
        self.perceptual_weight_d = perceptual_weight_d

    def padding_and_trimming(self, x, x_rec):
        _, _, H, W = x.size()
        x_rec = F.pad(x_rec, (15, 15, 15, 15), mode='replicate')
        x = F.pad(x, (15, 15, 15, 15), mode='replicate')
        _, _, H_pad, W_pad = x.size()
        top = random.randrange(0, 16)
        bottom = H_pad - random.randrange(0, 16)
        left = random.randrange(0, 16)
        right = W_pad - random.randrange(0, 16)
        x_rec = F.interpolate(x_rec[:, :, top:bottom, left:right],
        size=(H, W), mode='bicubic', align_corners=False)
        x = F.interpolate(x[:, :, top:bottom, left:right],
        size=(H, W), mode='bicubic', align_corners=False)
        return  x, x_rec

    def forward(self, reconstructions, inputs):
        inputs, reconstructions = self.padding_and_trimming(inputs, reconstructions)
        p_loss, p_loss_d  = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous()).chunk(2)
        # = self.perceptual_loss(compressed.contiguous(), inputs.contiguous())
        #  = torch.nn.functional.mse_loss(inputs.contiguous() , reconstructions.contiguous())
        if self.metric == ms_ssim:
            distortion = self.metric(reconstructions.contiguous(), inputs.contiguous(), data_range=2)
            rec_loss = 1 - distortion
        else:
            rec_loss = self.metric(reconstructions.contiguous(), inputs.contiguous())
            rec_loss_d = self.metric(compressed.contiguous(), inputs.contiguous())
            # distortion =  255**2 * out["mse_loss"]
        percptual_loss = rec_loss.mean() + self.perceptual_weight * p_loss.mean()
        percptual_loss_d = rec_loss_d.mean() + self.perceptual_weight_d * p_loss_d.mean()
        return rec_loss, p_loss, p_loss_d, percptual_loss, percptual_loss_d