import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v2.model_v2 import build_model_v2  # noqa: E402


def _load_any(ckpt_path: str, device: torch.device):
    ck = torch.load(ckpt_path, map_location=device)
    model = build_model_v2(pretrained=False).to(device)

    img_size = 224

    if isinstance(ck, dict) and "model_state" in ck:
        model.load_state_dict(ck["model_state"], strict=True)
        img_size = int(ck.get("img_size", 224))
    elif isinstance(ck, dict):
        model.load_state_dict(ck, strict=True)
    else:
        raise RuntimeError("Неизвестный формат чекпоинта")

    model.eval()
    return model, img_size


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to v2 checkpoint (.pt)")
    ap.add_argument("--img", required=True, help="Path to X-ray image (jpg/png)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"ckpt not found: {args.ckpt}")
    if not os.path.exists(args.img):
        raise FileNotFoundError(f"image not found: {args.img}")

    model, img_size = _load_any(args.ckpt, device)

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    with Image.open(args.img) as im:
        im = im.convert("RGB")
    x = tfm(im).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        lf = out["fracture"].item()
        lp = out["projection"].item()
        lh = out["hardware"].item()

        pf = sigmoid(lf)
        pp = sigmoid(lp)
        ph = sigmoid(lh)

    pred_f = int(pf >= 0.5)
    pred_p = int(pp >= 0.5)   # 1=S, 0=D
    pred_h = int(ph >= 0.5)

    proj_label = "S" if pred_p == 1 else "D"

    print("=== v2 inference ===")
    print(f"Fracture : prob={pf:.4f} | pred={pred_f}")
    print(f"Projection: prob={pp:.4f} | pred={proj_label} (S=1, D=0)")
    print(f"Hardware : prob={ph:.4f} | pred={pred_h}")


if __name__ == "__main__":
    main()

