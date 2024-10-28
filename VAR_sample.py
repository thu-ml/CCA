# Include VAR repo as a library
import sys
sys.path.append("./VAR")

################## 1. Download checkpoints and build models
import os
import os.path as osp
import torch, torchvision
import random
from tqdm import tqdm
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from VAR.models import VQVAE, build_vae_var
from PIL import Image
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str, default='path/to/your/var.pth')
parser.add_argument("--vae_ckpt", type=str, default='./vae_ch160v4096z32.pth')
parser.add_argument("--cfg", type=float, default=1.0)
parser.add_argument("--depth", type=int, default=16)
parser.add_argument("--sample_dir", type=str, default="./samples")

args = parser.parse_args()

MODEL_DEPTH = args.depth    # TODO: =====> please specify MODEL_DEPTH <=====
assert MODEL_DEPTH in {16, 20, 24, 30}


# download checkpoint
vae_ckpt, var_ckpt = args.vae_ckpt, args.ckpt_path


# build vae, var
patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if 'vae' not in globals() or 'var' not in globals():
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
    )

# load checkpoints
vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
vae.eval(), var.eval()
for p in vae.parameters(): p.requires_grad_(False)
for p in var.parameters(): p.requires_grad_(False)
print(f'prepare finished.')

############################# 2. Sample with classifier-free guidance

# set args
seed = 1 #@param {type:"number"}
num_sampling_steps = 250 #@param {type:"slider", min:0, max:1000, step:1}
cfg = args.cfg #@param {type:"slider", min:1, max:10, step:0.1}
more_smooth = False # True for more smooth output

# seed
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# run faster
tf32 = True
torch.backends.cudnn.allow_tf32 = bool(tf32)
torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
torch.set_float32_matmul_precision('high' if tf32 else 'highest')

path_parts = args.ckpt_path.replace(".pth", "").replace(".pt", "").split('/')
ckpt_string_name = f"{path_parts[-2]}-{path_parts[-1]}"

folder_name = f"d{MODEL_DEPTH}-{ckpt_string_name}-" \
                f"cfg-{args.cfg}-seed-{seed}-mode-{args.cfg_increase}"
sample_folder_dir = f"{args.sample_dir}/{folder_name}"
os.makedirs(sample_folder_dir, exist_ok=True)

# sample
B = 25
for img_cls in tqdm(range(1000)):
    for i in range(50 // B):
        label_B = torch.tensor([img_cls] * 25, device=device)
        # B = len(class_labels)
        # label_B: torch.LongTensor = torch.tensor(class_labels, device=device)
        with torch.inference_mode():
            with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
                # recon_B3HW = var.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=cfg, top_k=900, top_p=0.96, g_seed=seed + i, more_smooth=more_smooth)
                recon_B3HW = var.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=cfg, top_k=900, top_p=0.96, more_smooth=more_smooth)
            bchw = recon_B3HW.permute(0, 2, 3, 1).mul_(255).cpu().numpy()
        bchw = bchw.astype(np.uint8)
        for j in range(B):
            img = PImage.fromarray(bchw[j])
            img.save(f"{sample_folder_dir}/{(img_cls * 50 + i * B + j):06d}.png")

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    samples = np.random.permutation(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path
create_npz_from_sample_folder(sample_folder_dir)

