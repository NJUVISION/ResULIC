<!-- ## Ultra Lowrate Image Compression with Semantic Residual Coding and Compression-aware Diffusion -->

<h1 align="center"> Ultra Lowrate Image Compression with Semantic Residual Coding and Compression-aware Diffusion </h1>

<p align="center">
  Anle&nbsp;Ke<sup>1</sup> &ensp; <b>&middot;</b> &ensp; 
  Xu&nbsp;Zhang<sup>1</sup> &ensp; <b>&middot;</b> &ensp; 
  Tong&nbsp;Chen<sup>1</sup> &ensp; <b>&middot;</b> &ensp; 
  Ming&nbsp;Lu<sup>1</sup>&ensp; <b>&middot;</b> &ensp; 
  Chao&nbsp;Zhou<sup>2</sup>&ensp; <b>&middot;</b> &ensp; 
  Jiawen&nbsp;Gu<sup>2</sup>&ensp; <b>&middot;</b> &ensp;
  Zhan&nbsp;Ma<sup>1</sup>&ensp; </b> &ensp;
</p>

<p align="center">
  <sup>1</sup> Nanjing University &emsp; <sup>2</sup>Kuaishou Technology&emsp; 
</p>

<p align="center">
  <a href="https://njuvision.github.io/ResULIC/">🌐 Project Page</a> &ensp;
  <a href="https://arxiv.org/abs/2505.08281">📃 Paper</a> &ensp;
</p>

<p align="center">
    <img src="./fig/arc.png" style="border-radius: 15px"><br>
</p>

## :book: Table Of Contents
- [✨ Visual Results](#visual_results)
- [⏳ Train](#computer-train)
- [😀 Inference](#inference)
- [🌊 TODO](#todo)
- [❤ Acknowledgement](#acknowledgement)
- [🙇‍ Citation](#cite)


## ⚙️ Environment Setup

```bash
- conda create -n ResULIC python=3.10
- conda activate ResULIC
- pip install -r requirements.txt
```

## <a name="Visual Results"></a>✨ Visual Results
<p align="center">
    <img src="./fig/visual.png" style="border-radius: 15px"><br>
</p>

## <a name="train"></a>⏳ Train

**Note:** The numbers in the yaml filenames (e.g., `1_1_1`) represent $\lambda_{\text{diffusion}}$, $\lambda_{\text{mse}}$, and $\lambda_{\text{bpp}}$ respectively.

#### Stage 1: Initial Training

1. **Download Pretrained Model**  
   Download the pretrained **Stable Diffusion v2.1** model into the `./weight` directory:
   ```bash
   wget https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt --no-check-certificate -P ./weight

2. **Modify the configuration file**`./configs/train_zc_eps.yaml` and `./configs/model/stage1/xx.yaml` accordingly.

3. Start training.
   ```
   bash stage1.sh 
   ```
#### Stage 2: 

1. Modify the configuration file `./configs/train_stage2.yaml` and `./configs/model/stage2/xx.yaml` accordingly.

2. Start training.
   ```
   bash stage2.py 
   ```

## <a name="inference"></a>😀 Inference
 
1. Download the pre-trained models (We provide 3 low-bitrate models for testing, and other bitrate models can be trained by referring to the training method):

    | Bitrate   | Link(used config)       |
    | --------- | ---------- |
    | 0.03 bpp  | [ResULIC/configs/model/stage2/1_1_3/cldm_eps_300_ddim.yaml](https://modelscope.cn/models/kangle/SRIC-6/resolve/master/step%3D84999.ckpt) |
    | 0.01 bpp  | [ResULIC/configs/model/stage2/1_1_12/cldm_eps_400_ddim.yaml](https://modelscope.cn/models/kangle/ResULIC_add_steps400_0.09/resolve/master/step%3D199999.ckpt) |
    | 0.002 bpp | [ResULIC/configs/model/stage2/1_1_24/cldm_eps_600_ddim.yaml](https://modelscope.cn/models/kangle/ResULIC_add_steps600_0.0023/resolve/master/step%3D199999.ckpt) |

   
**Note:** `add_steps` and `Q` are only used to distinguish the testing image settings from the corresponding model settings. These parameters do not affect the results: `add_steps` can be aligned with the value in the config file, and `Q` corresponds to the trained $\lambda$. For the three provided models, the default `ddim_steps` is 3. If you use your own trained model, it is recommended to set `ddim_steps` to a divisor of `add_steps`. For example, when `add_steps=500`, `ddim_steps` could be 2, 4, 5...

1. W/o Srr, W/o Pfo.

   ```
   CUDA_VISIBLE_DEVICES=2 python inference_win.py \
    --ckpt xx \
    --config /xx/xx.yaml \
    --output xx/ \
    --ddim_steps 3 \
    --ddim_eta 0 \
    --Q x.0 \
    --add_steps x00
   ```

2. W/ Srr, W/o Pfo.
   ```
    CUDA_VISIBLE_DEVICES=2 python inference_res.py \
    --ckpt xx \
    --config /xx/xx.yaml \
    --output xx/ \
    --ddim_steps 3 \
    --ddim_eta 0 \
    --Q x.0 \
    --add_steps x00
    ```

3. W/ Srr, W/ Pfo.
   ```
    CUDA_VISIBLE_DEVICES=2 python inference_res_pfo.py \
    --ckpt xx \
    --config /xx/xx.yaml \
    --output xx/ \
    --ddim_steps 3 \
    --ddim_eta 0 \
    --Q x.0 \
    --add_steps x00
    ```

## <a name="todo"></a>🌊 TODO
- [x] Release code
- [x] Release quantitative metrics （👾The quantitative metrics for ResULIC presented in our paper can be found in [indicator](/indicator).）
- [x] Release pretrained models

## <a name="acknowledgement">❤ Acknowledgement
This work is based on [ControlNet](https://github.com/lllyasviel/ControlNet), [ControlNet-XS](https://github.com/vislearn/ControlNet-XS), [DiffEIC](https://github.com/huai-chang/DiffEIC), and [ELIC](https://github.com/JiangWeibeta/ELIC), thanks to their invaluable contributions.

## <a name="cite"></a>🙇‍ Citation


If you find our work useful, please consider citing:

```bibtex
@inproceedings{Ke2025resulic,
               author = {Ke, Anle and Zhang, Xu and Chen, Tong and Lu, Ming and Zhou, Chao and Gu, Jiawen and Ma, Zhan},
               title = {Ultra Lowrate Image Compression with Semantic Residual Coding and Compression-aware Diffusion},
               booktitle = {International Conference on Machine Learning},
               year = {2025}
               }
```
