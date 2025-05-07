import json
import time
from typing import List, Tuple
import os
import pandas as pd
from argparse import ArgumentParser, Namespace
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
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
from gpt import get_residual_caption, get_image_caption
# from qwen import get_residual_caption, get_image_caption
# from sensechat import get_residual_caption, get_image_caption
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

    decoded_img = model.decode_first_stage(compressed_img)
    decoded_img = torch.clamp((decoded_img + 1.0) / 2.0, min=0.0, max=1.0)
    compress_caption = get_image_caption(decoded_img)
    print(f"compress_caption: {compress_caption}")

    residual_caption = get_residual_caption(prompt, compress_caption)
    print(f"residual_caption: {residual_caption}")

    start_time = time.time()
    latent_x = sampler.stochastic_encode(compressed_img)
    # prompt2, best_ids, embedding, best_rec = prompt_optmizer.optimize_prompt(clip_model, model, clip_preprocess, args, model.device, compressed_img, latent_x, args.type, args.Q, target_images=img, prompt = prompt)
    encode_time = time.time() - start_time              
    print(f'prompt_time: {encode_time:.4f} seconds')

    prompt_ids_res = tokenizer.encode(residual_caption)
    best_ids_res = torch.tensor([prompt_ids_res], device=model.device)
    encoded_data_res, cdf_res, unique_chars_res = arithmetic_encode(best_ids_res)
    decoded_text_res = arithmetic_decode(encoded_data_res, cdf_res, unique_chars_res)
    decoded_text_res = torch.tensor(list(map(int, decoded_text_res.split(',')))).unsqueeze(0).to(model.device)
    decoded_text_res = decode_ids(decoded_text_res, tokenizer)[0]

    prompt_ids = tokenizer.encode(prompt)
    best_ids = torch.tensor([prompt_ids], device=model.device)
    encoded_data, cdf, unique_chars = arithmetic_encode(best_ids)
    decoded_text = arithmetic_decode(encoded_data, cdf, unique_chars)
    decoded_text = torch.tensor(list(map(int, decoded_text.split(',')))).unsqueeze(0).to(model.device)
    decoded_text = decode_ids(decoded_text, tokenizer)[0]

    print("Prompt:", decoded_text)
    print("Residual Prompt:", decoded_text_res)

    num_pixels = n_samples * height * width
    text_bpp = (len(encoded_data) * 8) / num_pixels
    text_bpp_res = (len(encoded_data_res) * 8) / num_pixels
    total_bpp = text_bpp + bpp
    total_bpp_res = text_bpp_res + bpp

    cond = {
        "c_latent": [compressed_img],
        "c_crossattn": [model.get_learned_conditioning([decoded_text + ', ' +  a_prompt] * n_samples)]
    }
    cond_res = {
        "c_latent": [compressed_img],
        "c_crossattn": [model.get_learned_conditioning([decoded_text_res + ', ' +  a_prompt] * n_samples)]
    }

    un_cond = {"c_latent": [compressed_img], "c_crossattn": [model.get_learned_conditioning([n_prompt] * n_samples)]}

    
    with torch.no_grad():
        # sampler: DDIMSampler
        start_time = time.time()
        samples = sampler.decode(latent_x, cond, unconditional_guidance_scale=args.scale,
                                            unconditional_conditioning=un_cond)
        end_time = time.time()
        print(f'sample_time: {end_time - start_time:.2f}s')

        start_time = time.time()
        samples_res = sampler.decode(latent_x, cond_res, unconditional_guidance_scale=args.scale,
                                            unconditional_conditioning=un_cond)
        end_time = time.time()
        print(f'sample_time: {end_time - start_time:.2f}s')
        
        x_samples = model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
        preds = [x_samples[i] for i in range(n_samples)]

        x_samples_res = model.decode_first_stage(samples_res)
        x_samples_res = torch.clamp((x_samples_res + 1.0) / 2.0, min=0.0, max=1.0)
        x_samples_res = (einops.rearrange(x_samples_res, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
        preds_res = [x_samples_res[i] for i in range(n_samples)]
    
    # best_rec = preds
    return preds, preds_res, bpp, text_bpp, text_bpp_res, total_bpp, total_bpp_res, compress_caption, residual_caption


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--color_fix_type", 
        type=str, 
        default="wavelet", 
        choices=["wavelet", "adain", "none"])
    parser.add_argument("--ckpt", default='**', type=str, help="Full checkpoint path")
    parser.add_argument("--config", default='**', type=str, help="Model config path")
    parser.add_argument("--json_file_path", type=str, default="**.json")
    parser.add_argument("--input", type=str, default= '/workspace/test/PRO2.0/Kodak', help="Path to input images")
    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddpm", "ddim"])
    # parser.add_argument("--steps", default=30, type=int)
    parser.add_argument("--scale", default=2.5, type=int)
    parser.add_argument("--excel", type=str, default='/workspace/test/ProSrc/kodak_caption/kodak_blip.xlsx', help="Path to Excel file containing prompts")
    parser.add_argument("--output", type=str, default='results_win_res_15/', help="Path to save results")
    parser.add_argument("--ddim_steps",type=int,default=3,help="number of ddim sampling steps",)
    parser.add_argument("--ddim_eta",type=float,default=0.0,help="ddim eta (eta=0.0 corresponds to deterministic sampling",)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--Q",type=float,default=4,help="")
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
    dists_metric_2 = DeepImageStructureTextureSimilarity().to(args.device)

    with open(args.json_file_path, 'r') as json_file:
        prompt_data = json.load(json_file)

    bpps = []
    text_bpps = []
    text_bpps_res = []
    total_bpps = []
    total_bpps_res = []
    lpips_scores = []
    psnr_scores = []
    psnr_scores_res = []
    msssim_scores = []
    msssim_scores_res = []
    img_results = [] 
    df = pd.read_excel(args.excel)
    assert os.path.isdir(args.input)
    print(f"Sampling {args.ddim_steps} steps using {args.sampler} sampler")
    # args_clip = Namespace()

    for i in range(24):
        file_name = f'kodim{str(i+1).zfill(2)}.png'
        file_path = os.path.join('/workspace/test/ProSrc/Kodak', file_name)

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

        image_index = f"kodim{str(i+1).zfill(2)}"
        
        for item in prompt_data:
            if item["image_index"] == image_index:
                prompt = item["original_caption"]

        # prompt = get_image_caption(file_path)
        # prompt = df.loc[i, 'original']
        print(f"prompt: {prompt}")

        preds, pred_res, bpp, text_bpp, text_bpp_res, total_bpp, total_bpp_res, com_caption, res_caption = process(
            model, [x], args, sampler=sampler,
            stream_path=stream_path, prompt=prompt,
        )
        pred = preds[0]
        pred_res = pred_res[0]

        bpps.append(bpp)
        text_bpps.append(text_bpp)
        text_bpps_res.append(text_bpp_res)
        total_bpps.append(total_bpp)
        total_bpps_res.append(total_bpp_res)
        # Remove padding
        pred = pred[:img.height, :img.width, :]

        # Save prediction
        Image.fromarray(pred).save(save_path)

        sample_path2 = os.path.join(parent_path, f"kodak_{args.Q}_{args.type}_{args.scale}_{args.ddim_steps}_{args.ddim_eta}_add{args.add_steps}/residual")
        os.makedirs(sample_path2, exist_ok=True)
        # base_count = len(os.listdir(sample_path2))/
        sample_path2 = os.path.join(sample_path2, f'{stem}.png')
        pred_res = pred_res[:img.height, :img.width, :]
        Image.fromarray(pred_res).save(sample_path2)

        # Convert images to tensors
        img_tensor = torch.tensor(np.array(img) / 255.0).permute(2, 0, 1).unsqueeze(0).to(args.device).float()
        pred_tensor = torch.tensor(pred / 255.0).permute(2, 0, 1).unsqueeze(0).to(args.device).float()
        pred_tensor_res = torch.tensor(pred_res / 255.0).permute(2, 0, 1).unsqueeze(0).to(args.device).float()

        # Calculate metrics
        lpips_score = lpips_metric(img_tensor * 2 - 1, pred_tensor * 2 - 1, normalize=False).item()
        lpips_metric_2(pred_tensor_res, img_tensor)
        dists_score = dists_metric.update(img_tensor, pred_tensor)
        dists_metric_2.update(img_tensor, pred_tensor_res)
        psnr_score = calculate_psnr_pt(img_tensor, pred_tensor, crop_border=0).mean().item()
        psnr_score_res = calculate_psnr_pt(img_tensor, pred_tensor_res, crop_border=0).mean().item()
        msssim_score = calculate_msssim_pt(img_tensor, pred_tensor)
        msssim_score_res = calculate_msssim_pt(img_tensor, pred_tensor_res)
        image_metrics = {
            "image_index": stem,
            "original_caption": prompt,
            "compressed_caption": com_caption,
            "residual_caption": res_caption,
            "image_bpp": bpp,
            "text_bpp": text_bpp,
            "text_bpp_res": text_bpp_res,
            "total_bpp": total_bpp,
            "total_bpp_res": total_bpp_res,
            "LPIPS_loss": lpips_score,
            "MS-SSIM": msssim_score,
            "MS-SSIM_res": msssim_score_res,
            "PSNR": psnr_score,
            "PSNR_res": psnr_score_res
        }

        lpips_scores.append(lpips_score)
        psnr_scores.append(psnr_score)
        psnr_scores_res.append(psnr_score_res)
        msssim_scores.append(msssim_score)
        msssim_scores_res.append(msssim_score_res)
        img_results.append(image_metrics)
        
        print(f"Saved to {save_path}, bpp: {bpp}, text_bpp:{text_bpp}, total_bpp:{total_bpp}, LPIPS: {lpips_score}, PSNR: {psnr_score}, MS-SSIM: {msssim_score}")

    # Calculate averages
    avg_bpp = sum(bpps) / len(bpps)
    avg_text_bpp = sum(text_bpps) / len(text_bpps)
    avg_text_bpp_res = sum(text_bpps_res) / len(text_bpps)
    avg_total_bpp = sum(total_bpps) / len(total_bpps)
    avg_total_bpp_res = sum(total_bpps_res) / len(total_bpps)
    avg_lpips = sum(lpips_scores) / len(lpips_scores)
    avg_psnr = sum(psnr_scores) / len(psnr_scores)
    avg_psnr_res = sum(psnr_scores_res) / len(psnr_scores_res)
    avg_msssim = sum(msssim_scores) / len(msssim_scores)
    avg_msssim_res = sum(msssim_scores_res) / len(msssim_scores_res)
    similarity_score = float(dists_metric.compute())
    similarity_score_res = float(dists_metric_2.compute())
    lpips_total = float(lpips_metric_2.compute())

    print(f"\nAverage Metrics:\nBPP: {avg_bpp}\nText_BPP:{avg_text_bpp}\nTotal_BPP:{avg_total_bpp}\nLPIPS: {avg_lpips}\nPSNR: {avg_psnr}\nMS-SSIM: {avg_msssim}\nDISTS: {similarity_score}\nLPIPS_2: {lpips_total}")
    
    results = {
    "AVG_Lpips": f"{avg_lpips}",
    "AVG_LPIPS_res": f"{lpips_total}",
    "AVG_DISTS": f"{similarity_score}",
    "AVG_DISTS_res": f"{similarity_score_res}",
    "AVG_MS-SSIM": f"{avg_msssim}",
    "AVG_MS-SSIM_res": f"{avg_msssim_res}",
    "AVG_PSNR": f"{avg_psnr}dB",  
    "AVG_PSNR_res": f"{avg_psnr_res}dB",
    "AVG_test_bpp": f"{avg_text_bpp} bpp",
    "AVG_text_bpp_res": f"{avg_text_bpp_res} bpp",
    "AVG_image_bpp": f"{avg_bpp} bpp",
    "AVG_total_bpp": f"{avg_total_bpp} bpp", 
    "AVG_total_bpp_res": f"{avg_total_bpp_res} bpp"
}
    
    output_file = os.path.join(parent_path, f"kodak_{args.Q}_{args.type}_{args.scale}_{args.ddim_steps}_{args.ddim_eta}_add{args.add_steps}/kodak.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    output_file2 = os.path.join(parent_path, f"kodak_{args.Q}_{args.type}_{args.scale}_{args.ddim_steps}_{args.ddim_eta}_add{args.add_steps}/kodak_detailed.json")
    with open(output_file2, 'w') as f:
        json.dump(img_results, f, indent=4)

if __name__ == "__main__":
    main()
