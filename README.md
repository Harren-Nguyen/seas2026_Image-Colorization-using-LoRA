# Grayscale Image Colorization for Central Vietnam using CNN and LoRA
This project uses a CNN architecture based on the paper Colorful Image Colorization by Richard Zhang et al. (ECCV 2016), combined with Low-Rank Adaptation (LoRA), to colorize historical grayscale images of Central Vietnam - a region with many important historical sites and artifacts. By keeping the L channel and predicting only a and b in the Lab color space, the model reproduces exactly the original colors of ancient architecture while saving over 90% of trainable parameters.

Experiments on 1,000+ heritage images show that LoRA fine-tuning significantly improves PSNR, SSIM, and MSE metrics, generating vivid, natural colorizations for complex historical architecture. The project supports historical restoration, digital exhibitions, tourism, and education. Future work includes expanding to nationwide datasets and exploring Diffusion Models with ControlNet.

![Logo GitHub](https://github.com/user-attachments/assets/f96d3dd1-cd26-4259-8913-0bb3b7c84ab3)

# Technical Overview & Architecture
__1) Base Backbone (ECCVColorizer)__

The backbone replicates Zhang et al.'s architecture:   

- Input: CIELAB $L$-channel tensor $(B, 1, H, W)$ normalized to $[-0.5, 0.5]$.   

- Feature Extractor: 8 convolutional blocks incorporating dilated convolutions (dilations 2) to increase the receptive field without downsampling.   

- Output: Soft-decoded $a^*b^*$ color channel prediction $(B, 2, H/4, W/4)$ scaled to $[-110, 110]$.   

__2) Low-Rank Adaptation (nn.Conv2d)__

Rather than updating all $100\%$ of backbone parameters, a low-rank adapter branch wraps every nn.Conv2d layer:
<div align="center">
  
$$\text{Output} = W_{\text{base}}(x) + \frac{\alpha}{r} \cdot \Big(B \cdot A\Big)(x)$$

</div>

- $W_{\text{base}}$ remains completely frozen.   
- $A \in \mathbb{R}^{r \times C_{\text{in}} \times k_H \times k_W}$ is initialized with Kaiming Uniform.   
- $B \in \mathbb{R}^{C_{\text{out}} \times r \times 1 \times 1}$ is initialized to zero.
- Trainable Parameters: Reduces trainable weights to $< 2\%$ of the original model.

# Installation & Setup
__Prerequisites__

Python $\ge 3.9$

PyTorch with CUDA support

__Environment Setup__
```Bash
# Clone the repository:
git clone https://github.com/Harren-Nguyen/Image-Colorization-using-LoRA.git
cd Image-Colorization-using-LoRA

# Create virtual environment:
python -m venv venv
source venv\Scripts\activate

# Install dependencies:
pip install requirements.txt
```

# Training & Usage

__1)__ Run `lora_train.py` by specifying your dataset directory. Pretrained base weights are downloaded automatically if not found locally.
```Python
lora_train.py --train-dir /path/to/your/dataset/images/ --epochs 10 --batch-size 8 --lr 1e-4 --rank 9 --alpha 9.0
```

__Key Training Options__

- `--epochs`: Number of full training passes over the dataset (default: 100).
- `--batch-size`: Number of images processed per training step (default: 8).
- `--lr`: Learning rate - step size for updating weights without disrupting pretrained features (default: 1e-4).
- `--train-dir`: Directory containing RGB training images (e.g., .jpg, .png, .webp).
- `--rank`: Rank $r$ of the LoRA adapter (default: 9).
- `--alpha`: Scaling factor $\alpha$ for LoRA updates (default: 9.0).
- `--lora-save`: Path to export adapter weights (default: checkpoints/lora_weights.pth).

__2)__ Run `lora_inference.py` to colorize grayscale images in the `input` directory using trained LoRA weights. The colorized outputs will be exported to `output`.
```Python
lora_inference.py --input-dir
```

# Credits & Academic Acknowledgments

This research project was conducted under the direct supervision of our research mentors and in collaboration with our project team, within the SEAS 2026 Summer School (Quang Tri, Vietnam):

- Research Supervisors/Mentors: Nguyen Phuc Luong, Nguyen Thi Nhu Quynh, Nguyen Phu Vinh, M.Sc.
- Research Contributors: Nguyen Vinh Trong, Tran Hoang To, Tran Cong Dat, Tran Duc Phat, Nguyen Ngoc Tram Anh, Luong Thi Chi Lan

# Citation and License

If you use this codebase or the underlying colorization models in your research, please cite the original papers by Richard Zhang et al., as requested by the authors:

```bibtex
@inproceedings{zhang2016colorful,
  title={Colorful Image Colorization},
  author={Zhang, Richard and Isola, Phillip and Efros, Alexei A},
  booktitle={ECCV},
  year={2016}
}

@article{zhang2017real,
  title={Real-Time User-Guided Image Colorization with Learned Deep Priors},
  author={Zhang, Richard and Zhu, Jun-Yan and Isola, Phillip and Geng, Xinyang and Lin, Angela S and Yu, Tianhe and Efros, Alexei A},
  journal={ACM Transactions on Graphics (TOG)},
  volume={36},
  number={4},
  year={2017},
  publisher={ACM}
}

This project is open-source and distributed under the MIT License. See the LICENSE file for more information.
