import json
import time
from typing import List, Tuple
import os
import pandas as pd
from argparse import ArgumentParser, Namespace
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import numpy as np
import torch
import einops
import pytorch_lightning as pl
from PIL import Image
from omegaconf import OmegaConf

from ldm.xformers_state import disable_xformers
from model.spaced_sampler import SpacedSampler
# from model.ddim_sampler import DDIMSampler
from model.ddim_zc import DDIMSampler
from utils.image import pad
from utils.common import instantiate_from_config, load_state_dict
from utils.file import list_image_files, get_file_name_parts
from utils.metrics import calculate_msssim_pt, calculate_psnr_pt, LPIPS
import prompt_inversion.test_optim_zc as prompt_optmizer
import prompt_inversion.open_clip as open_clip 
from nn_indices import arithmetic_decode, arithmetic_encode
from neuralcompression.metrics import DeepImageStructureTextureSimilarity
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
)
from model.color import wavelet_reconstruction, adaptive_instance_normalization
# from open_clip import OpenCLIPModel, Preprocess

def decode_ids(input_ids, tokenizer, by_token=False):
    input_ids = input_ids.detach().cpu().numpy()
    texts = []
    if by_token:
        for input_ids_i in input_ids:
            curr_text = []
            for tmp in input_ids_i:
                curr_text.append(tokenizer.decode([tmp]))
            texts.append('|'.join(curr_text))
    else:
        for input_ids_i in input_ids:
            texts.append(tokenizer.decode(input_ids_i))
    return texts
# @torch.no_grad()
def process(model, imgs, args, sampler, stream_path, prompt):
   
    n_samples = len(imgs)
    
    control = torch.tensor(np.stack(imgs) / 255.0, dtype=torch.float32, device=model.device).clamp_(0, 1)
    control = einops.rearrange(control, "n h w c -> n c h w").contiguous()
    
    
    tokenizer = open_clip.tokenizer._tokenizer

    height, width = control.size(-2), control.size(-1)

    bpp = model.apply_condition_compress(control, stream_path, height, width)

    a_prompt = 'best quality, extremely detailed'
    n_prompt = 'oil painting, cartoon, blurring, dirty, messy, low quality, frames, deformed, lowres, over-smooth.'

    
    compressed_img = model.apply_condition_decompress(stream_path)
    start_time = time.time()
    latent_x = sampler.stochastic_encode(compressed_img)
    # prompt2, best_ids, embedding, best_rec = prompt_optmizer.optimize_prompt(clip_model, model, clip_preprocess, args, model.device, compressed_img, latent_x, args.type, args.Q, target_images=img, prompt = prompt)
    decode_time = time.time() - start_time              
    print(f'decode_time: {decode_time:.4f} seconds')

    time_start = time.time()
    prompt_ids = tokenizer.encode(prompt)
    best_ids = torch.tensor([prompt_ids], device=model.device)
    
    encoded_data, cdf, unique_chars = arithmetic_encode(best_ids)
    print(f"text_encode_time: {(time.time() - time_start):.4f} seconds")
    time_start = time.time()
    decoded_text = arithmetic_decode(encoded_data, cdf, unique_chars)
    decoded_text = torch.tensor(list(map(int, decoded_text.split(',')))).unsqueeze(0).to(model.device)

    decoded_text = decode_ids(decoded_text, tokenizer)[0]
    print(f"text_decode_time: {(time.time() - time_start):.4f} seconds")
    print("Prompt:", decoded_text)
    num_pixels = n_samples * height * width
    text_bpp = (len(encoded_data) * 8) / num_pixels
    total_bpp = text_bpp + bpp

    cond = {
        "c_latent": [compressed_img],
        "c_crossattn": [model.get_learned_conditioning([decoded_text + ', ' +  a_prompt] * n_samples)]
    }
    un_cond = {"c_latent": [compressed_img], "c_crossattn": [model.get_learned_conditioning([n_prompt] * n_samples)]}

    
    with torch.no_grad():
        sampler: DDIMSampler
        start_time = time.time()
        samples = sampler.decode(latent_x, cond, unconditional_guidance_scale=args.scale,
                                            unconditional_conditioning=un_cond)
        end_time = time.time()
        print(f'sample_time: {end_time - start_time:.2f}s')
        
        x_samples = model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        # if args.color_fix_type == "adain":
        #     color_img = model.decode_first_stage(compressed_img)
        #     color_img = torch.clamp((color_img + 1.0) / 2.0, min=0.0, max=1.0)
        #     x_samples = adaptive_instance_normalization(x_samples, color_img)
        # elif args.color_fix_type == "wavelet":
        #     color_img = model.decode_first_stage(compressed_img)
        #     color_img = torch.clamp((color_img + 1.0) / 2.0, min=0.0, max=1.0)
        #     x_samples = wavelet_reconstruction(x_samples, color_img)
        # x_samples = x_samples.clamp(0, 1)
        
        x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
        
        preds = [x_samples[i] for i in range(n_samples)]
    
    # best_rec = preds
    return preds, bpp, text_bpp, total_bpp


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--color_fix_type", 
        type=str, 
        default="wavelet", 
        choices=["wavelet", "adain", "none"])
    parser.add_argument("--ckpt", default='/workspace/SRIC/logs_new/1_1_3_add_xs_eps_300_lpips/lightning_logs/version_2/checkpoints/step=79999.ckpt', type=str, help="Full checkpoint path")
    parser.add_argument("--config", default='/workspace/SRIC/configs/model/lpips/cldm_eps_300_ddim.yaml', type=str, help="Model config path")
    
    parser.add_argument("--input", type=str, default= '/workspace/SRIC/Kodak', help="Path to input images")
    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddpm", "ddim"])
    # parser.add_argument("--steps", default=30, type=int)
    parser.add_argument("--scale", default=2.5, type=int)
    parser.add_argument("--excel", type=str, default='/workspace/SRIC/kodak_caption/kodak_blip.xlsx', help="Path to Excel file containing prompts")
    parser.add_argument("--output", type=str, default='results_win_gan/', help="Path to save results")
    parser.add_argument("--ddim_steps",type=int,default=3,help="number of ddim sampling steps",)
    parser.add_argument("--ddim_eta",type=float,default=0.0,help="ddim eta (eta=0.0 corresponds to deterministic sampling",)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--Q",type=float,default=3,help="")
    parser.add_argument("--add_steps",type=int,default=300,help="")
    parser.add_argument("--type",type=str,default="lpips")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed)
    
    if args.device == "cpu":
        disable_xformers()  

    model = instantiate_from_config(OmegaConf.load(args.config))
    load_state_dict(model, torch.load(args.ckpt, map_location="cuda"), strict=False)
    model.preprocess_model.update(force=True)
    model.freeze()
    model.to(args.device)

    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=args.ddim_steps, ddim_eta=args.ddim_eta, verbose=True)

    lpips_metric = LPIPS(net="alex").to(args.device)
    lpips_metric_2 = LearnedPerceptualImagePatchSimilarity(normalize=True).to(args.device)
    dists_metric = DeepImageStructureTextureSimilarity().to(args.device)
    
    bpps = []
    text_bpps = []
    total_bpps = []
    lpips_scores = []
    psnr_scores = []
    msssim_scores = []
    img_results = [] 
    df = pd.read_excel(args.excel)
    assert os.path.isdir(args.input)
    print(f"Sampling {args.ddim_steps} steps using {args.sampler} sampler")
    # args_clip = Namespace()

    for i in range(24):
        file_name = f'kodim{str(i+1).zfill(2)}.png'
        file_path = os.path.join('/workspace/SRIC/Kodak', file_name)

        img = Image.open(file_path).convert("RGB")
        x = pad(np.array(img), scale=64)

        save_path = os.path.join(args.output, file_name)
        parent_path, stem, _ = get_file_name_parts(save_path)
        stream_parent_path = os.path.join(parent_path, f'kodak_{args.Q}_{args.type}_{args.scale}_{args.ddim_steps}_{args.ddim_eta}_add{args.add_steps}/data')
        save_path = os.path.join(parent_path, f"kodak_{args.Q}_{args.type}_{args.scale}_{args.ddim_steps}_{args.ddim_eta}_add{args.add_steps}/{stem}.png")
        stream_path = os.path.join(stream_parent_path, f"{stem}")

        os.makedirs(parent_path, exist_ok=True)
        os.makedirs(stream_parent_path, exist_ok=True)

        # Get prompt for the current image
        prompt = df.loc[i, 'original']

        preds, bpp, text_bpp, total_bpp = process(
            model, [x], args, sampler=sampler,
            stream_path=stream_path, prompt=prompt,
        )
        pred = preds[0]

        bpps.append(bpp)
        text_bpps.append(text_bpp)
        total_bpps.append(total_bpp)
        # Remove padding
        pred = pred[:img.height, :img.width, :]

        # Save prediction
        Image.fromarray(pred).save(save_path)


        # Convert images to tensors
        img_tensor = torch.tensor(np.array(img) / 255.0).permute(2, 0, 1).unsqueeze(0).to(args.device).float()
        pred_tensor = torch.tensor(pred / 255.0).permute(2, 0, 1).unsqueeze(0).to(args.device).float()

        # Calculate metrics
        lpips_score = lpips_metric(img_tensor * 2 - 1, pred_tensor * 2 - 1, normalize=False).item()
        lpips_metric_2(pred_tensor, img_tensor)
        dists_score = dists_metric.update(img_tensor, pred_tensor)
        psnr_score = calculate_psnr_pt(img_tensor, pred_tensor, crop_border=0).mean().item()
        msssim_score = calculate_msssim_pt(img_tensor, pred_tensor)
        image_metrics = {
            "image_index": stem,
            "image_bpp": bpp,
            "text_bpp": text_bpp,
            "total_bpp": total_bpp,
            "LPIPS_loss": lpips_score,
            "MS-SSIM": msssim_score,
            "PSNR": psnr_score
        }

        lpips_scores.append(lpips_score)
        psnr_scores.append(psnr_score)
        msssim_scores.append(msssim_score)
        img_results.append(image_metrics)
        
        print(f"Saved to {save_path}, bpp: {bpp}, text_bpp:{text_bpp}, total_bpp:{total_bpp}, LPIPS: {lpips_score}, PSNR: {psnr_score}, MS-SSIM: {msssim_score}")

    # Calculate averages
    avg_bpp = sum(bpps) / len(bpps)
    avg_text_bpp = sum(text_bpps) / len(text_bpps)
    avg_total_bpp = sum(total_bpps) / len(total_bpps)
    avg_lpips = sum(lpips_scores) / len(lpips_scores)
    avg_psnr = sum(psnr_scores) / len(psnr_scores)
    avg_msssim = sum(msssim_scores) / len(msssim_scores)
    similarity_score = float(dists_metric.compute())
    lpips_total = float(lpips_metric_2.compute())

    print(f"\nAverage Metrics:\nBPP: {avg_bpp}\nText_BPP:{avg_text_bpp}\nTotal_BPP:{avg_total_bpp}\nLPIPS: {avg_lpips}\nPSNR: {avg_psnr}\nMS-SSIM: {avg_msssim}\nDISTS: {similarity_score}\nLPIPS_2: {lpips_total}")
    
    results = {
    "AVG_Lpips": f"{avg_lpips}",
    "AVG_MS-SSIM": f"{avg_msssim}",
    "AVG_PSNR": f"{avg_psnr}dB",  
    "AVG_test_bpp": f"{avg_text_bpp} bpp",
    "AVG_image_bpp": f"{avg_bpp} bpp",
    "AVG_total_bpp": f"{avg_total_bpp} bpp", 
    "AVG_DISTS": f"{similarity_score}",
    "AVG_LPIPS_2": f"{lpips_total}"
}
    
    output_file = os.path.join(parent_path, f"kodak_{args.Q}_{args.type}_{args.scale}_{args.ddim_steps}_{args.ddim_eta}_add{args.add_steps}/kodak.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    output_file2 = os.path.join(parent_path, f"kodak_{args.Q}_{args.type}_{args.scale}_{args.ddim_steps}_{args.ddim_eta}_add{args.add_steps}/kodak_detailed.json")
    with open(output_file2, 'w') as f:
        json.dump(img_results, f, indent=4)

if __name__ == "__main__":
    main()
