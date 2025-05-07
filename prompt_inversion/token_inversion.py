import math
import re
import os
import random
from einops import rearrange
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from statistics import mean
import copy
import lpips
import time
import json
from pytorch_msssim import ms_ssim
from typing import Any, Mapping
import torch.nn.functional as F
from torchvision import transforms
from pytorch_lightning import seed_everything
from scipy.optimize import fmin_l_bfgs_b
from ldm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder
from model.ddim_zc import DDIMSampler
from ldm.modules.diffusionmodules.util_2 import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, extract_into_tensor
# from semantic_segmentation.mit_semseg.lib import nn
from . import open_clip
# from ..text_code import huffman_encoding, huffman_decoding
# import open_clip

import torch

from sentence_transformers.util import (semantic_search, 
                                        dot_score, 
                                        cos_sim,
                                        euclidean_sim,
                                        normalize_embeddings)


def read_json(filename: str) -> Mapping[str, Any]:
    """Returns a Python dict representation of JSON object at input file."""
    with open(filename) as fp:
        return json.load(fp)

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def nn_project(curr_embeds, embedding_layer, tokenizer, print_hits=False):
    with torch.no_grad():
        bsz,seq_len,emb_dim = curr_embeds.shape
        
        # Using the sentence transformers semantic search which is 
        # a dot product exact kNN search between a set of 
        # query vectors and a corpus of vectors
        curr_embeds = curr_embeds.reshape((-1,emb_dim))
        curr_embeds = normalize_embeddings(curr_embeds) # queries


        embedding_matrix = embedding_layer.weight
        embedding_matrix = normalize_embeddings(embedding_matrix)
        
        hits = semantic_search(curr_embeds, embedding_matrix, 
                                query_chunk_size=curr_embeds.shape[0], 
                                top_k=1,
                                score_function=euclidean_sim)


        if print_hits:
            all_hits = []
            for hit in hits:
                all_hits.append(hit[0]["score"])
            print(f"mean hits:{mean(all_hits)}")


        nn_indices = torch.tensor([hit[0]["corpus_id"] for hit in hits], device=curr_embeds.device)
        nn_indices = nn_indices.reshape((bsz,seq_len))

        
        projected_embeds = embedding_layer(nn_indices)


    return projected_embeds, nn_indices



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


def get_target_feature(model, preprocess, tokenizer_funct, device, target_images=None, target_prompts=None):
    if target_images is not None:
        with torch.no_grad():
            # curr_images = [preprocess(i).unsqueeze(0) for i in target_images]
            curr_images = preprocess(target_images).unsqueeze(0).to(device)
            #curr_images = torch.cat(curr_images).to(device)
            all_target_features = model.encode_image(curr_images)
    else:
        texts = tokenizer_funct(target_prompts).to(device)
        all_target_features = model.encode_text(texts)

    return all_target_features


def initialize_prompt(tokenizer, token_embedding, opt, device, initial_prompt_text=None):

    if initial_prompt_text is not None:
        # Encode the specified initial prompt text
        initial_prompt_text = re.sub(r'[^\w\s]', ' ', initial_prompt_text)
        prompt_ids = tokenizer.encode(initial_prompt_text)
        # Convert prompt_ids to tensor and move to device
        prompt_len = opt.prompt_len
        if len(prompt_ids) > prompt_len:
            # Truncate if the prompt is too long
            prompt_ids = prompt_ids[:prompt_len]
        elif len(prompt_ids) < prompt_len:
            # Pad with a special padding token (e.g., tokenizer.pad_token_id)
            padding_id = 0  # Use 0 as the padding token ID if pad_token_id is not available
            padding = [padding_id] * (prompt_len - len(prompt_ids))
            prompt_ids.extend(padding)
        
        # Convert prompt_ids to tensor and move to device
        prompt_ids = torch.tensor([prompt_ids], device=device)

        # prompt_ids = torch.tensor([prompt_ids], device=device)
        # prompt_len = prompt_ids.size(1)
    else:
        # Randomly generate prompt embeddings as fallback
        prompt_len = opt.prompt_len
        prompt_ids = torch.randint(len(tokenizer.encoder), (opt.prompt_bs, prompt_len)).to(device)

    prompt_embeds = token_embedding(prompt_ids).detach()
    print(prompt_embeds.shape)

    prompt_embeds.requires_grad = True
    
    # initialize the template
    template_text = "{}"
    padded_template_text = template_text.format(" ".join(["<start_of_text>"] * prompt_len))
    dummy_ids = tokenizer.encode(padded_template_text)

    # -1 for optimized tokens
    dummy_ids = [i if i != 49406 else -1 for i in dummy_ids]
    dummy_ids = [49406] + dummy_ids + [49407]
    dummy_ids += [0] * (77 - len(dummy_ids))
    dummy_ids = torch.tensor([dummy_ids] * opt.prompt_bs).to(device)

    # for getting dummy embeds; -1 won't work for token_embedding
    tmp_dummy_ids = copy.deepcopy(dummy_ids)
    tmp_dummy_ids[tmp_dummy_ids == -1] = 0
    dummy_embeds = token_embedding(tmp_dummy_ids).detach()
    dummy_embeds.requires_grad = False
    
    return prompt_embeds, dummy_embeds, dummy_ids, prompt_len

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()

def get_loss(pred, target, mean=True):
        if mean:
            loss = torch.nn.functional.mse_loss(target, pred)
        else:
            loss = torch.nn.functional.mse_loss(target, pred, reduction='none')


        return loss

def optimize_prompt_loop(model, diffusion, preprocess, tokenizer, token_embedding, compressed_img, latent_x, target_images, all_target_features, opt, device, prompt, type, Q):
    torch.manual_seed(opt.seed)
    loss_fn_alex = lpips.LPIPS(net='alex',eval_mode=True)
    loss_fn_alex.to(device)
    sampler = DDIMSampler(diffusion)
    opt_iters = opt.iter
    lr = opt.lr
    # opt.eval_step = 50


    a_prompt = 'best quality, extremely detailed'
    n_prompt = 'oil painting, cartoon, blurring, dirty, messy, low quality, frames, deformed, lowres, over-smooth.'

    # ori_embeddings = diffusion.get_learned_conditioning([prompt + ', ' + a_prompt])

    target_image = transforms.ToTensor()(target_images).unsqueeze(0).to(device)
    bsz = target_image.shape[0]

    prompt_embeds, dummy_embeds, dummy_ids, prompt_len = initialize_prompt(tokenizer, token_embedding, opt, device, prompt)
    p_bs, p_len, p_dim = prompt_embeds.shape

    input_optimizer = torch.optim.AdamW([prompt_embeds], lr=opt.lr, weight_decay=opt.weight_decay)

    if type == 'psnr' or type =='ssim':
        best_loss = -999
        eval_loss = -9999
    if type == 'lpips':
        best_loss = 999
        eval_loss = 9999
    best_text = ""
    best_embeds = None
    compressed_img.to(device)
    ddim_timesteps = make_ddim_timesteps(ddim_discr_method="uniform", num_ddim_timesteps=opt.ddim_steps, num_ddpm_timesteps=diffusion.add_steps, verbose=True)
    
    sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=True)

    x = diffusion.get_first_stage_encoding(diffusion.encode_first_stage(target_image)).detach()

    # control_res = compressed_img.detach().clone()
    uc =  {"c_latent": [compressed_img], "c_crossattn": [diffusion.get_learned_conditioning([n_prompt])]}

    for name, param in diffusion.named_parameters():
        if "model.diffusion_model" or "control_model" in name:
            param.requires_grad = True  
        else:
            param.requires_grad = False
    dummy_ids = dummy_ids.to(device)
    
    for step in range(opt_iters):
        with torch.cuda.amp.autocast():
        # forward projection
        # projected_embeds, projected_embeds2, nn_indices, nn_indices2 = nn_project(prompt_embeds, token_embedding, tokenizer, compress_features)
        
            projected_embeds, nn_indices = nn_project(prompt_embeds, token_embedding, tokenizer)

            #tmp_embeds = copy.deepcopy(prompt_embeds)
            tmp_embeds = prompt_embeds.detach().clone()
            tmp_embeds.data = projected_embeds.data
            tmp_embeds.requires_grad = True
            #prompt_embeds = projected_embeds.data
            #prompt_embeds.requires_grad = True
            # padding and repeat
            #padded_embeds = copy.deepcopy(dummy_embeds)
            # padded_embeds = dummy_embeds.detach().clone()
            padded_embeds2 = dummy_embeds.detach().clone()
            # padded_embeds[:, 1:prompt_len+1] = tmp_embeds#
            # padded_embeds = padded_embeds.repeat(opt.batch_size, 1, 1)
            # padded_dummy_ids = dummy_ids.repeat(opt.batch_size, 1)
            padded_embeds2[dummy_ids == -1] = tmp_embeds.reshape(-1, p_dim)
            
            logits_per_image, logits_per_text, text_embeddings = model.forward_text_embedding(padded_embeds2, dummy_ids, all_target_features)
            cosim_scores = logits_per_image
            loss = 1 - cosim_scores.mean()
            loss = loss * opt.loss_weight


            index = np.random.randint(0, len(ddim_timesteps), size=(bsz,))
            index = torch.tensor(index, device=device)      
            timesteps = ddim_timesteps[index.item()]
            # timesteps = torch.randint(400, 500, (bsz,), device=device)
            timesteps_tensor = torch.tensor([timesteps])
            timesteps = timesteps_tensor.long().to(device)

            # text_embeddings2 = text_embeddings.detach()
            
            # cond = {"c_latent": [compressed_img], "c_crossattn": [text_embeddings]}
            cond = {"c_latent": None, "c_crossattn": [text_embeddings]}

            # torch.manual_seed(42)
            noise = torch.randn_like(x)
            # with torch.no_grad():
            x_noisy = diffusion.q_sample(x_start=x, t=timesteps, z_c=compressed_img, noise=noise)
            model_output = diffusion.apply_model(x_noisy, timesteps, cond)
            # pred_x0 = diffusion.predict_start_from_noise(x_noisy, t=timesteps, z_c=compressed_img, noise=model_output)
            loss_diffusion = get_loss(model_output, noise, mean=False).mean([1, 2, 3])
            # loss_diffusion = get_loss(pred_x0, x, mean=False).mean([1, 2, 3])
            loss_diffusion = loss_diffusion.mean()
            # print("time step: ", timesteps)
            # print("loss_diffusion: ", loss_diffusion)
            # loss_diffusion = torch.mean(torch.nn.functional.mse_loss(x, pred_x0)

            loss = loss + loss_diffusion * opt.loss_diffusion

            
        prompt_embeds.grad, = torch.autograd.grad(loss, [tmp_embeds])

        input_optimizer.step()
        input_optimizer.zero_grad()    
        
        curr_lr = input_optimizer.param_groups[0]["lr"]
        sample_path = f'./optimize_zc/{type}_{Q}'
        os.makedirs(sample_path, exist_ok=True)
        base_count = len(os.listdir(sample_path))

        if step % opt.eval_step == 0:
            
            decoded_text = decode_ids(nn_indices, tokenizer)[0]

            text_embeddings2 = diffusion.get_learned_conditioning([decoded_text+ ', ' + a_prompt ])

            cond_i = {"c_latent": [compressed_img], "c_crossattn": [text_embeddings2]}
            print(f"step: {step}, lr: {curr_lr}, cosim: {eval_loss:.3f}, best_cosim: {best_loss:.3f}, best prompt: {best_text},\
                  input prompt: {decoded_text}")
                
            with torch.no_grad():
                with diffusion.ema_scope():
                    torch.manual_seed(opt.seed)
                    diffusion.eval()
                    diffusion.control_scales =  ([1] * 13)

                    x_samples = sampler.decode(latent_x, cond_i, unconditional_guidance_scale=opt.scale,
                                                    unconditional_conditioning=uc)
                    pred_imgs = diffusion.decode_first_stage(x_samples)
                    pred_imgs = torch.clamp((pred_imgs + 1.0) / 2.0, min=0.0, max=1.0)
                    for pred_img in pred_imgs:
                        pred_img = 255. * rearrange(pred_img.cpu().numpy(), 'c h w -> h w c')
                        pred_img = Image.fromarray(pred_img.astype(np.uint8))
                        pred_img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                        base_count += 1
                        
                        if type == 'lpips':
                            eval_loss = torch.mean(loss_fn_alex(pred_imgs, target_image, normalize=True))

                        if type =='ssim':
                            eval_loss=compute_msssim(target_image, pred_imgs)
                        if type =='psnr':
                            eval_loss=compute_psnr(target_image, pred_imgs)

            print(f'type: {type}')
            if type =='ssim' or type =='psnr':
                if best_loss < eval_loss: #and universal_cosim_score < eval_loss:
                    best_loss = eval_loss
                    best_text = decoded_text
                    best_ids = nn_indices
                    best_rec = pred_imgs
                    best_embeds = text_embeddings 
                                        
            if type =='lpips':
                if best_loss > eval_loss: #and universal_cosim_score < eval_loss:
                    best_loss = eval_loss
                    best_text = decoded_text
                    best_ids = nn_indices
                    best_rec = pred_imgs
                    best_embeds = text_embeddings

    print(f"Best shot: consine similarity: {best_loss:.3f}")
    print(f"text: {best_text}")  

    return best_text,best_ids, best_embeds, best_rec

    
def optimize_prompt(model, diffusion, preprocess, opt, device, compressed_img, latent_x, type, Q, target_images=None,  prompt=None, target_prompts=None):
    #image_encoder=FrozenOpenCLIPImageEmbedder()
    token_embedding = model.token_embedding
    tokenizer = open_clip.tokenizer._tokenizer
    tokenizer_funct = open_clip.get_tokenizer(opt.clip_model)

    # get target features
    all_target_features = get_target_feature(model, preprocess, tokenizer_funct, device, target_images=target_images, target_prompts=target_prompts)

    learned_prompt, learned_ids, emb, best_rec = optimize_prompt_loop(model, diffusion, preprocess, tokenizer, token_embedding, compressed_img, latent_x, target_images, all_target_features, opt, device, prompt, type, Q)

    return learned_prompt, learned_ids, emb, best_rec


    

def measure_similarity(orig_images, images, ref_model, ref_clip_preprocess, device):
    with torch.no_grad():

        ori_batch = ref_clip_preprocess(orig_images).unsqueeze(0).to(device)

        gen_batch = ref_clip_preprocess(images).unsqueeze(0).to(device)
        
        ori_feat = ref_model.encode_image(ori_batch)
        gen_feat = ref_model.encode_image(gen_batch)
        
        ori_feat = ori_feat / ori_feat.norm(dim=1, keepdim=True)
        gen_feat = gen_feat / gen_feat.norm(dim=1, keepdim=True)
        
        return (ori_feat @ gen_feat.t()).mean().item()
    

def clip_cosine(orig_images, images, ref_model, ref_clip_preprocess, device):
    with torch.no_grad():
        ori_batch = [ref_clip_preprocess(i).unsqueeze(0) for i in orig_images]
        ori_batch = torch.cat(ori_batch).to(device)

        gen_batch = [ref_clip_preprocess(i).unsqueeze(0) for i in images]
        gen_batch = torch.cat(gen_batch).to(device)
        
        ori_feat = ref_model.encode_image(ori_batch)
        gen_feat = ref_model.encode_image(gen_batch)
        
        ori_feat = ori_feat / ori_feat.norm(dim=1, keepdim=True)
        gen_feat = gen_feat / gen_feat.norm(dim=1, keepdim=True)
        
        return (ori_feat @ gen_feat.t()).diag()