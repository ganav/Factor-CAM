# explain_posthoc_concepts.py
from __future__ import annotations
from pathlib import Path
import csv, numpy as np
import torch, torch.nn.functional as F
import torch_directml as dml
from torchvision import datasets, transforms
from PIL import Image, ImageFilter
import matplotlib
import numpy as np
from PIL import Image
import colorsys
import open_clip

# ---------- CONFIG ----------
MODEL_PATH = "artifacts/conceptcnn_full.pt"
OUT_DIR    = "artifacts/explain"
IMG_SIZE   = 224
LIMIT      = None         # e.g., 100 to test quickly
TOPK_RATIO = 0.20         # CAM top % kept as region
USE_CLIP   = False        # True to name shape/texture phrases (CPU by default)
CLIP_MODEL = "ViT-B-32"
CLIP_PRETR = "laion2b_s34b_b79k"
CLIP_DEVICE = "cpu"
SHAPES   = ["pointy triangular ear shape", "rounded ear shape", "elongated snout shape"]
TEXTURES = ["striped fur texture", "curly fur texture", "smooth short fur"]
# ----------------------------

DEV = dml.device(0)  # DirectML GPU for classifier

# ----- dataset loader (reuses your data directory) -----
def _ensure_dataset(root: Path) -> Path:
    """Import from your data_concepts.py at runtime if present; else minimal fallback."""
    try:
        from data_concepts import _ensure_dataset as _ens
        return _ens(root)
    except Exception:
        # Fallback: assumes you already downloaded 'cats_and_dogs_filtered'
        return root / "cats_and_dogs_filtered"

def load_val(img_size=224):
    data_dir = _ensure_dataset(Path("data"))
    tfm = transforms.Compose([
        transforms.Resize(int(img_size*1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    ds = datasets.ImageFolder(data_dir/"validation", transform=tfm)
    return ds, ds.classes

# ----- utilities -----
IMAGENET_MEAN = np.array([0.485,0.456,0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229,0.224,0.225], dtype=np.float32)

def to_uint8(x3chw):
    x = x3chw.detach().cpu().permute(1,2,0).numpy()
    x = x * IMAGENET_STD + IMAGENET_MEAN
    x = np.clip(x, 0, 1)
    return (x*255).astype(np.uint8)

def overlay(img_rgb, heat, alpha=0.45):
    
    heat = np.clip(heat, 0, 1)
    color = matplotlib.colormaps.get_cmap("jet")(heat)[...,:3]
    color = (color*255).astype(np.uint8)
    return (alpha*color + (1-alpha)*img_rgb).astype(np.uint8)

def cam_topk_bbox(M_up: np.ndarray, keep_ratio=0.2):
    flat = M_up.flatten()
    thr = np.quantile(flat, 1.0-keep_ratio)
    ys, xs = np.where(M_up >= thr)
    if len(xs)==0:  # fallback full image
        return (0,0,M_up.shape[1], M_up.shape[0])
    return (xs.min(), ys.min(), xs.max()-xs.min()+1, ys.max()-ys.min()+1)

def paste(base, crop, bbox):
    x,y,w,h = bbox
    out = base.copy()
    out[y:y+h, x:x+w, :] = crop
    return out

def from_uint8(u8):
    x = u8.astype(np.float32)/255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x).permute(2,0,1).float()

def dominant_color_name(cam_crop_rgb):
    
    img = cam_crop_rgb.reshape(-1,3).astype(np.float32)/255.0
    hsv = np.array([colorsys.rgb_to_hsv(*px) for px in img])
    S = hsv[:,1]; V = hsv[:,2]
    mask = (S > 0.2) & (V > 0.2)
    if mask.sum() < 50:
        if V.mean()<0.25: return "black"
        if V.mean()>0.75: return "white"
        return "gray"
    H = hsv[mask,0]; hdeg = (H*360.0)
    # simple bins (tune as you like)
    if ((hdeg>=20)&(hdeg<45)).mean()>0.4: return "yellow"
    if ((hdeg>=315)|(hdeg<20)).mean()>0.4: return "red-brown"
    if ((hdeg>=75)&(hdeg<170)).mean()>0.4: return "green"
    if ((hdeg>=170)&(hdeg<255)).mean()>0.4: return "cyan-blue"
    if ((hdeg>=255)&(hdeg<315)).mean()>0.4: return "purple"
    return "brown"

# ----- factor ablations (test-time) -----
def make_noColor(crop_rgb):
    return np.array(Image.fromarray(crop_rgb).convert("L").convert("RGB"))

def make_noTexture(crop_rgb, radius=8):
    return np.array(Image.fromarray(crop_rgb).filter(ImageFilter.GaussianBlur(radius=radius)))

def make_noShape(crop_rgb, tiles=8):
    """
    Breaks shape by shuffling grid cells and resizing source cells to fit
    destination cells so sizes always match (no broadcasting errors).
    """


    H, W = crop_rgb.shape[:2]
    ys = np.linspace(0, H, tiles + 1, dtype=int)
    xs = np.linspace(0, W, tiles + 1, dtype=int)

    # all destination cells
    dst_cells = [(ys[i], ys[i+1], xs[j], xs[j+1])
                 for i in range(tiles) for j in range(tiles)]
    # random source indices
    src_cells = dst_cells.copy()
    np.random.shuffle(src_cells)

    out = np.zeros_like(crop_rgb)
    for (y0d,y1d,x0d,x1d), (y0s,y1s,x0s,x1s) in zip(dst_cells, src_cells):
        src = crop_rgb[y0s:y1s, x0s:x1s, :]
        dh, dw = (y1d - y0d), (x1d - x0d)
        # resize source block to exactly match destination block size
        src_resized = np.array(Image.fromarray(src).resize((dw, dh), Image.BILINEAR))
        out[y0d:y1d, x0d:x1d, :] = src_resized
    return out

def factor_probs(model, x_chw, M_up_np, pred_class, DEV):
    # bbox on CAM
    bbox = cam_topk_bbox(M_up_np, keep_ratio=TOPK_RATIO)

    # baseline logit
    with torch.no_grad():
        L0 = model(x_chw.unsqueeze(0).to(DEV))[0][0, pred_class].item()

    # RGB image
    img = to_uint8(x_chw)
    x,y,w,h = bbox
    crop = img[y:y+h, x:x+w, :]

    # counterfactual crops
    crop_nc = make_noColor(crop)         # remove color
    crop_nt = make_noTexture(crop, 8)    # remove texture
    crop_ns = make_noShape(crop, 8)      # break shape

    # paste back
    img_nc = paste(img, crop_nc, bbox)
    img_nt = paste(img, crop_nt, bbox)
    img_ns = paste(img, crop_ns, bbox)

    # logits for counterfactuals
    def logit(u8):
        xt = from_uint8(u8).unsqueeze(0).to(DEV)
        with torch.no_grad():
            return model(xt)[0][0, pred_class].item()

    Lc = logit(img_nc); Lt = logit(img_nt); Ls = logit(img_ns)
    d_color   = max(0.0, L0 - Lc)
    d_texture = max(0.0, L0 - Lt)
    d_shape   = max(0.0, L0 - Ls)
    S = d_color + d_texture + d_shape + 1e-6
    return (d_shape/S, d_color/S, d_texture/S), (d_shape, d_color, d_texture), bbox

# ----- optional CLIP phrase naming -----
def clip_rank_phrases(clip_model, preprocess, phrases, cam_crop_pil, device="cpu"):
    
    with torch.no_grad():
        tok = open_clip.get_tokenizer(CLIP_MODEL)(phrases).to(device)
        te = clip_model.encode_text(tok); te = te/te.norm(dim=-1,keepdim=True)
        img = preprocess(cam_crop_pil).unsqueeze(0).to(device)
        ie = clip_model.encode_image(img); ie = ie/ie.norm(dim=-1,keepdim=True)
        sim = (ie @ te.T)[0]
        top = int(sim.argmax().item())
        return phrases[top], float(sim[top].item())

def main():
    out_dir = Path(OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)
    ds, class_names = load_val(IMG_SIZE)
    # load classifier on DirectML
    model = torch.load(MODEL_PATH, map_location="cpu").to(DEV).eval()
    print("MODEL DEVICE:", next(model.parameters()).device)

    # optional CLIP
    if USE_CLIP:

        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETR, device=CLIP_DEVICE
        )
        clip_model.eval()

    csv_path = out_dir/"posthoc_explanations.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([
            "idx","img_path","true","pred","pred_prob",
            "p_shape","p_color","p_texture",
            "d_shape","d_color","d_texture",
            "shape_phrase","shape_score","texture_phrase","texture_score",
            "color_name","overlay_path","bbox_x","bbox_y","bbox_w","bbox_h"
        ])

        count=0
        for i,(x,y) in enumerate(ds):
            if LIMIT is not None and count>=LIMIT: break
            x_in = x.unsqueeze(0).to(DEV)
            with torch.no_grad():
                ylog, _, M = model(x_in)
                prob = torch.softmax(ylog,dim=1)[0]
                pred = int(prob.argmax().item())
                pcls = float(prob[pred].item())
                M_up = F.interpolate(M, size=(x.shape[1], x.shape[2]),
                                     mode="bilinear", align_corners=False)[0,0].cpu().numpy()

            # save CAM overlay
            img_rgb = to_uint8(x)
            over = overlay(img_rgb, M_up)
            overlay_path = out_dir / f"val_{i:05d}_pred-{class_names[pred]}.png"
            Image.fromarray(over).save(overlay_path)

            # factor probabilities via ablations
            (ps, pc, pt), (ds_, dc_, dt_), bbox = factor_probs(model, x, M_up, pred, DEV)

            # crop for CLIP naming and color name
            x0,y0,w0,h0 = bbox
            crop_rgb = img_rgb[y0:y0+h0, x0:x0+w0, :]
            crop_pil = Image.fromarray(crop_rgb)
            if USE_CLIP:
                shape_phrase, shape_score     = clip_rank_phrases(clip_model, preprocess, SHAPES, crop_pil, CLIP_DEVICE)
                texture_phrase, texture_score = clip_rank_phrases(clip_model, preprocess, TEXTURES, crop_pil, CLIP_DEVICE)
            else:
                shape_phrase, shape_score = "", ""
                texture_phrase, texture_score = "", ""
            color_name = dominant_color_name(crop_rgb)

            w.writerow([
                i, ds.samples[i][0], class_names[y], class_names[pred], f"{pcls:.6f}",
                f"{ps:.4f}", f"{pc:.4f}", f"{pt:.4f}",
                f"{ds_:.4f}", f"{dc_:.4f}", f"{dt_:.4f}",
                shape_phrase, f"{shape_score:.4f}" if shape_score!="" else "",
                texture_phrase, f"{texture_score:.4f}" if texture_score!="" else "",
                color_name, str(overlay_path), x0,y0,w0,h0
            ])
            count+=1

    print(f"[posthoc] overlays → {OUT_DIR}, CSV → {csv_path}")

if __name__ == "__main__":
    main()
