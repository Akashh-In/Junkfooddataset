"""
=============================================================
  Junk Food YOLO Object Detection — Full Training & Eval
=============================================================
  Dataset repo : https://github.com/Akashh-In/Junkfooddataset
  Model        : YOLOv8n / s / m  (configurable)
  Outputs      : runs/detect/junkfood_exp/
                  ├── weights/best.pt
                  ├── confusion_matrix.png
                  ├── PR_curve.png
                  ├── F1_curve.png
                  ├── results.png          (loss + metric curves)
                  └── val_predictions/     (annotated val images)

  Usage:
      1.  pip install ultralytics matplotlib seaborn scikit-learn tqdm
      2.  Clone dataset:
              git clone https://github.com/Akashh-In/Junkfooddataset.git
      3.  python train_junkfood_yolo.py
=============================================================
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import os
import sys
import shutil
import yaml
import random
import argparse
from pathlib import Path

import numpy as np
import matplotlib

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
CFG = {
    # ---------- paths ----------
    "dataset_root"  : ".",                 # dataset root next to this script
    "data_yaml"     : "./data.yaml",
    "output_dir"    : "./runs/detect/junkfood_exp",

    # ---------- training ----------
    "model"         : "yolov8n.pt",   # yolov8n / yolov8s / yolov8m
    "epochs"        : 50,
    "imgsz"         : 640,
    "batch"         : 16,
    "device"        : "",             # "" = auto-select (GPU if available)
    "workers"       : 4,

    # ---------- misc ----------
    "conf_thresh"   : 0.25,
    "iou_thresh"    : 0.45,
    "seed"          : 42,
    "patience"      : 10,
    "cache"         : False,
    "val"           : True,
    "fraction"      : 1.0,
    "max_eval_images": 0,
    "skip_extras"   : False,
}

# ── Argument overrides ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Junk Food YOLO Trainer")
parser.add_argument("--epochs",  type=int,   default=CFG["epochs"])
parser.add_argument("--model",   type=str,   default=CFG["model"])
parser.add_argument("--batch",   type=int,   default=CFG["batch"])
parser.add_argument("--imgsz",   type=int,   default=CFG["imgsz"])
parser.add_argument("--device",  type=str,   default=CFG["device"])
parser.add_argument("--patience", type=int,  default=CFG["patience"],
                    help="Early stop patience (epochs)")
parser.add_argument("--cache", action="store_true",
                    help="Cache images in RAM for faster training")
parser.add_argument("--no_val", action="store_true",
                    help="Disable per-epoch validation (faster)")
parser.add_argument("--fast", action="store_true",
                    help="Speed-focused settings to fit a tight time budget")
parser.add_argument("--fraction", type=float, default=CFG["fraction"],
                    help="Fraction of training data to use (0-1)")
parser.add_argument("--max_eval_images", type=int, default=CFG["max_eval_images"],
                    help="Limit images used for custom confusion matrix (0=all)")
parser.add_argument("--skip_extras", action="store_true",
                    help="Skip extra plots/confusion matrix/sample predictions")
parser.add_argument("--show", action="store_true",
                    help="Show plots in GUI windows (requires Tk backend)")
parser.add_argument("--eval_only", action="store_true",
                    help="Skip training; only run evaluation on best.pt")
args = parser.parse_args()

CFG["epochs"] = args.epochs
CFG["model"]  = args.model
CFG["batch"]  = args.batch
CFG["imgsz"]  = args.imgsz
CFG["device"] = args.device
CFG["patience"] = args.patience
CFG["cache"] = args.cache
CFG["val"] = not args.no_val
CFG["fraction"] = max(0.0, min(1.0, args.fraction))
CFG["max_eval_images"] = max(0, args.max_eval_images)
CFG["skip_extras"] = args.skip_extras
EVAL_ONLY     = args.eval_only
SHOW_PLOTS    = args.show

if args.fast:
    CFG["epochs"] = min(CFG["epochs"], 15)
    CFG["imgsz"] = min(CFG["imgsz"], 512)
    CFG["patience"] = min(CFG["patience"], 5)
    CFG["val"] = False
    CFG["cache"] = True
    if CFG["fraction"] >= 1.0:
        CFG["fraction"] = 0.5

if SHOW_PLOTS:
    try:
        matplotlib.use("TkAgg")
    except Exception as exc:
        SHOW_PLOTS = False
        matplotlib.use("Agg")
        print(f"[WARN] GUI backend unavailable ({exc}). Falling back to file output.")
else:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


def _pick_existing(candidates, must_be_dir=False):
    """Return first existing candidate path, else None."""
    for p in candidates:
        p = Path(p)
        if must_be_dir and p.is_dir():
            return p.resolve()
        if (not must_be_dir) and p.is_file():
            return p.resolve()
    return None


def _looks_like_dataset_root(path):
    """Return True when the path contains the expected YOLO split folders."""
    path = Path(path)
    return (
        path.is_dir()
        and (path / "images" / "train").is_dir()
        and (path / "images" / "val").is_dir()
        and (path / "labels" / "train").is_dir()
        and (path / "labels" / "val").is_dir()
    )


def normalize_cfg_paths():
    """Resolve config paths robustly (cwd + script-relative fallbacks)."""
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()

    # dataset root
    ds_raw = Path(CFG["dataset_root"])
    ds_candidates = [
        cwd,
        script_dir,
        cwd / ds_raw,
        script_dir / ds_raw,
        cwd / "Junkfooddataset",
        script_dir / "Junkfooddataset",
    ]
    ds_found = next((p.resolve() for p in ds_candidates if _looks_like_dataset_root(p)), None)
    if ds_found is not None:
        CFG["dataset_root"] = str(ds_found)
    else:
        CFG["dataset_root"] = str(script_dir.resolve())

    # data yaml
    yaml_raw = Path(CFG["data_yaml"])
    yaml_candidates = [
        yaml_raw,
        cwd / yaml_raw,
        script_dir / yaml_raw,
        Path(CFG["dataset_root"]) / "data.yaml",
        cwd / "data.yaml",
        script_dir / "data.yaml",
    ]
    yaml_found = _pick_existing(yaml_candidates, must_be_dir=False)
    if yaml_found is not None:
        CFG["data_yaml"] = str(yaml_found)
    else:
        CFG["data_yaml"] = str((script_dir / yaml_raw).resolve())

    # output dir
    out_raw = Path(CFG["output_dir"])
    CFG["output_dir"] = str((out_raw if out_raw.is_absolute() else script_dir / out_raw).resolve())

    print("\n[PATHS]")
    print(f"  cwd          : {cwd}")
    print(f"  script_dir   : {script_dir}")
    print(f"  dataset_root : {CFG['dataset_root']}")
    print(f"  data_yaml    : {CFG['data_yaml']}")
    print(f"  output_dir   : {CFG['output_dir']}")


def normalize_data_yaml():
    """Normalize data.yaml into a resolved copy with valid paths."""
    data_yaml = Path(CFG["data_yaml"])
    if not data_yaml.exists():
        script_dir = Path(__file__).resolve().parent
        root = Path(CFG["dataset_root"])
        checked = [
            data_yaml,
            script_dir / "data.yaml",
            root / "data.yaml",
            Path.cwd() / "data.yaml",
        ]
        msg = "\n".join([f"    - {p}" for p in checked])
        sys.exit(
            f"[ERROR] data.yaml not found. Checked:\n{msg}\n"
            "  -> Fix CFG['data_yaml'] or place data.yaml in dataset root."
        )

    with open(data_yaml) as f:
        meta = yaml.safe_load(f) or {}

    class_names = meta.get("names", [])
    if isinstance(class_names, dict):
        class_names = [class_names[k] for k in sorted(class_names)]
    elif not isinstance(class_names, list):
        class_names = list(class_names) if class_names else []
    meta["names"] = class_names
    if "nc" not in meta:
        meta["nc"] = len(class_names)

    default_splits = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
    }
    for split, default_path in default_splits.items():
        if not meta.get(split):
            meta[split] = default_path

    base_dir = data_yaml.parent
    candidates = []
    if meta.get("path"):
        root_path = Path(meta["path"])
        if not root_path.is_absolute():
            root_path = (base_dir / root_path).resolve()
        candidates.append(root_path)
    candidates.append(Path(CFG["dataset_root"]))
    candidates.append(base_dir)

    fixed_root = next((p.resolve() for p in candidates if _looks_like_dataset_root(p)), None)
    if fixed_root is None:
        fixed_root = Path(CFG["dataset_root"]).resolve()

    meta["path"] = str(fixed_root)
    CFG["dataset_root"] = str(fixed_root)

    for split, default_path in default_splits.items():
        split_val = meta.get(split)
        split_path = Path(split_val)
        if split_path.is_absolute():
            if not split_path.exists():
                candidate = fixed_root / default_path
                if candidate.exists():
                    meta[split] = default_path
        else:
            if not (fixed_root / split_path).exists():
                candidate = fixed_root / default_path
                if candidate.exists():
                    meta[split] = default_path

    resolved_yaml = OUT / "data_resolved.yaml"
    with open(resolved_yaml, "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    CFG["data_yaml"] = str(resolved_yaml)
    print(f"  data_yaml (resolved): {CFG['data_yaml']}")
    return meta


normalize_cfg_paths()

OUT = Path(CFG["output_dir"])
OUT.mkdir(parents=True, exist_ok=True)
normalize_data_yaml()

# ═════════════════════════════════════════════════════════════════════════════
# 1. DATASET SANITY CHECK
# ═════════════════════════════════════════════════════════════════════════════
def check_dataset():
    """Verify dataset structure and report class counts."""
    print("\n" + "="*60)
    print("  DATASET SANITY CHECK")
    print("="*60)

    data_yaml = Path(CFG["data_yaml"])
    if not data_yaml.exists():
        script_dir = Path(__file__).resolve().parent
        root = Path(CFG["dataset_root"])
        checked = [
            data_yaml,
            script_dir / "data.yaml",
            root / "data.yaml",
            Path.cwd() / "data.yaml",
        ]
        msg = "\n".join([f"    - {p}" for p in checked])
        sys.exit(
            f"[ERROR] data.yaml not found. Checked:\n{msg}\n"
            "  → Fix CFG['data_yaml'] or place data.yaml in dataset root."
        )

    with open(data_yaml) as f:
        meta = yaml.safe_load(f)

    class_names = meta.get("names", [])
    if isinstance(class_names, dict):
        class_names = [class_names[k] for k in sorted(class_names)]
    elif not isinstance(class_names, list):
        class_names = list(class_names) if class_names else []
    nc          = meta.get("nc", len(class_names))
    print(f"  Classes ({nc}): {class_names}")

    root = Path(CFG["dataset_root"])
    required_dirs = [
        root / "images" / "train",
        root / "images" / "val",
        root / "labels" / "train",
        root / "labels" / "val",
    ]
    missing = [p for p in required_dirs if not p.exists()]
    if missing:
        msg = "\n".join([f"    - {p}" for p in missing])
        sys.exit(
            f"[ERROR] Dataset folders not found. Missing:\n{msg}\n"
            "  -> Check data.yaml 'path' and local dataset layout."
        )

    # Validate split paths defined inside data.yaml (relative paths supported)
    yaml_base = Path(meta.get("path", data_yaml.parent))
    if not yaml_base.is_absolute():
        yaml_base = (data_yaml.parent / yaml_base).resolve()
    for split in ["train", "val", "test"]:
        split_path = meta.get(split)
        if split_path:
            p = Path(split_path)
            if not p.is_absolute():
                p = (yaml_base / p).resolve()
            if not p.exists():
                print(f"[WARN] data.yaml '{split}' path does not exist: {p}")

    for split in ["train", "val", "test"]:
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        if img_dir.exists():
            imgs   = list(img_dir.glob("*.*"))
            labels = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []
            print(f"  {split:5s} → {len(imgs):4d} images | {len(labels):4d} labels")
        else:
            print(f"[WARN] Missing images directory: {img_dir}")

    return class_names, nc


# ═════════════════════════════════════════════════════════════════════════════
# 2. TRAIN
# ═════════════════════════════════════════════════════════════════════════════
def train():
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("[ERROR] ultralytics not installed.\n  pip install ultralytics")

    print("\n" + "="*60)
    print("  TRAINING")
    print("="*60)

    model = YOLO(CFG["model"])
    results = model.train(
        data        = CFG["data_yaml"],
        epochs      = CFG["epochs"],
        imgsz       = CFG["imgsz"],
        batch       = CFG["batch"],
        device      = CFG["device"] or None,
        workers     = CFG["workers"],
        project     = str(OUT.parent),
        name        = OUT.name,
        exist_ok    = True,
        seed        = CFG["seed"],
        patience    = CFG["patience"],
        cache       = CFG["cache"],
        val         = CFG["val"],
        fraction    = CFG["fraction"],
        verbose     = True,
    )
    print(f"\n  ✔ Training complete → {OUT}")
    return model


# ═════════════════════════════════════════════════════════════════════════════
# 3. VALIDATE & COLLECT PREDICTIONS
# ═════════════════════════════════════════════════════════════════════════════
def validate(class_names):
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("[ERROR] ultralytics not installed.")

    best_pt = OUT / "weights" / "best.pt"
    if not best_pt.exists():
        sys.exit(f"[ERROR] best.pt not found at {best_pt}")

    print("\n" + "="*60)
    print("  VALIDATION")
    print("="*60)

    model = YOLO(str(best_pt))
    metrics = model.val(
        data     = CFG["data_yaml"],
        imgsz    = CFG["imgsz"],
        conf     = CFG["conf_thresh"],
        iou      = CFG["iou_thresh"],
        device   = CFG["device"] or None,
        project  = str(OUT),
        name     = "eval",
        exist_ok = True,
        verbose  = True,
    )

    # ── Print summary ────────────────────────────────────────────────────────
    print(f"\n  mAP@50      : {metrics.box.map50:.4f}")
    print(f"  mAP@50-95   : {metrics.box.map:.4f}")
    print(f"  Precision   : {metrics.box.mp:.4f}")
    print(f"  Recall      : {metrics.box.mr:.4f}")

    return model, metrics


# ═════════════════════════════════════════════════════════════════════════════
# 4. CUSTOM CONFUSION MATRIX  (from val label files + predictions)
# ═════════════════════════════════════════════════════════════════════════════
def build_confusion_matrix(class_names):
    """
    Run inference on every val image, match ground-truth vs predicted
    class, and draw a clean seaborn confusion matrix.
    """
    try:
        from ultralytics import YOLO
        from PIL import Image
    except ImportError:
        print("[WARN] Skipping custom confusion matrix (missing lib).")
        return

    best_pt  = OUT / "weights" / "best.pt"
    val_imgs = sorted((Path(CFG["dataset_root"]) / "images" / "val").glob("*.*"))
    val_lbls = Path(CFG["dataset_root"]) / "labels" / "val"

    if CFG["max_eval_images"] > 0 and len(val_imgs) > CFG["max_eval_images"]:
        val_imgs = val_imgs[:CFG["max_eval_images"]]

    if not val_imgs:
        print("[WARN] No val images found — skipping confusion matrix.")
        return

    model = YOLO(str(best_pt))
    nc    = len(class_names)

    y_true, y_pred = [], []

    print("\n  Building confusion matrix …")
    for img_path in tqdm(val_imgs, unit="img"):
        lbl_path = val_lbls / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        # ground-truth classes for this image
        with open(lbl_path) as f:
            gt_classes = [int(line.split()[0]) for line in f if line.strip()]

        if not gt_classes:
            continue

        # predictions
        results = model.predict(
            str(img_path),
            conf    = CFG["conf_thresh"],
            iou     = CFG["iou_thresh"],
            verbose = False,
        )
        pred_classes = results[0].boxes.cls.cpu().numpy().astype(int).tolist() \
                       if results[0].boxes is not None else []

        # simple matching: pair by order (majority-class fallback)
        # For a richer IoU-based matching, use ultralytics built-in CM.
        for gt in gt_classes:
            if pred_classes:
                y_true.append(gt)
                y_pred.append(pred_classes.pop(0))
            else:
                y_true.append(gt)
                y_pred.append(nc)          # "background" / missed

        for pc in pred_classes:            # extra false positives
            y_true.append(nc)
            y_pred.append(pc)

    # clip to valid range
    y_true = np.clip(y_true, 0, nc)
    y_pred = np.clip(y_pred, 0, nc)
    labels = list(range(nc + 1))
    tick_labels = class_names + ["background"]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(max(8, nc + 2), max(6, nc + 1)))
    sns.heatmap(
        cm,
        annot     = True,
        fmt       = "d",
        cmap      = "Blues",
        xticklabels = tick_labels,
        yticklabels = tick_labels,
        ax        = ax,
        linewidths = 0.5,
    )
    ax.set_xlabel("Predicted", fontsize=12, labelpad=10)
    ax.set_ylabel("Ground Truth", fontsize=12, labelpad=10)
    ax.set_title("Confusion Matrix — Junk Food Detection", fontsize=14, pad=14)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    save_path = OUT / "custom_confusion_matrix.png"
    plt.savefig(save_path, dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    print(f"  ✔ Confusion matrix saved → {save_path}")

    # ── per-class report ─────────────────────────────────────────────────────
    valid_mask = (np.array(y_true) < nc) & (np.array(y_pred) < nc)
    if valid_mask.sum() > 0:
        report = classification_report(
            np.array(y_true)[valid_mask],
            np.array(y_pred)[valid_mask],
            labels      = list(range(nc)),
            target_names = class_names,
            zero_division = 0,
        )
        print("\n  Per-class Classification Report:\n")
        print(report)
        with open(OUT / "classification_report.txt", "w") as f:
            f.write(report)


# ═════════════════════════════════════════════════════════════════════════════
# 5. TRAINING CURVES  (loss + mAP over epochs)
# ═════════════════════════════════════════════════════════════════════════════
def plot_training_curves():
    """Read results.csv from ultralytics and plot loss + metrics."""
    csv_path = OUT / "results.csv"
    if not csv_path.exists():
        print("[WARN] results.csv not found — skipping training curves.")
        return

    import csv

    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip(): float(v) for k, v in row.items() if v.strip()})

    if not rows:
        return

    epochs = [r.get("epoch", i + 1) for i, r in enumerate(rows)]

    def get(key):
        return [r.get(key, float("nan")) for r in rows]

    # ── Loss curves ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    loss_map = {
        "Box Loss"  : ("train/box_loss",   "val/box_loss"),
        "Class Loss": ("train/cls_loss",   "val/cls_loss"),
        "DFL Loss"  : ("train/dfl_loss",   "val/dfl_loss"),
    }
    for ax, (title, (tr_key, vl_key)) in zip(axes, loss_map.items()):
        ax.plot(epochs, get(tr_key), label="Train",      color="#1f77b4", lw=2)
        ax.plot(epochs, get(vl_key), label="Validation", color="#ff7f0e", lw=2,
                linestyle="--")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle("Training & Validation Loss", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(OUT / "loss_curves.png", dpi=150, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    print(f"  ✔ Loss curves saved → {OUT}/loss_curves.png")

    # ── mAP / Precision / Recall ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metric_map = {
        "Precision"  : "metrics/precision(B)",
        "Recall"     : "metrics/recall(B)",
        "mAP@50"     : "metrics/mAP50(B)",
    }
    colors = ["#2ca02c", "#d62728", "#9467bd"]
    for ax, (title, key), color in zip(axes, metric_map.items(), colors):
        vals = get(key)
        ax.plot(epochs, vals, color=color, lw=2)
        ax.fill_between(epochs, vals, alpha=0.15, color=color)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Validation Metrics over Epochs", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(OUT / "metric_curves.png", dpi=150, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    print(f"  ✔ Metric curves saved → {OUT}/metric_curves.png")


# ═════════════════════════════════════════════════════════════════════════════
# 6. PRECISION–RECALL CURVE  (manual, per-class)
# ═════════════════════════════════════════════════════════════════════════════
def plot_pr_curve(class_names):
    """
    If ultralytics saved PR data in the eval folder, re-plot it nicely.
    Otherwise, skip.
    """
    # Ultralytics saves confusion_matrix.csv and other data in the val dir.
    pr_src = OUT / "eval"
    if not pr_src.exists():
        return

    # ultralytics already saves PR_curve.png — just copy & annotate
    pr_img = pr_src / "PR_curve.png"
    if pr_img.exists():
        shutil.copy(pr_img, OUT / "PR_curve.png")
        print(f"  ✔ PR curve copied → {OUT}/PR_curve.png")

    f1_img = pr_src / "F1_curve.png"
    if f1_img.exists():
        shutil.copy(f1_img, OUT / "F1_curve.png")
        print(f"  ✔ F1 curve copied → {OUT}/F1_curve.png")

    cm_img = pr_src / "confusion_matrix.png"
    if cm_img.exists():
        shutil.copy(cm_img, OUT / "confusion_matrix_ultralytics.png")
        print(f"  ✔ Ultralytics confusion matrix copied → {OUT}/confusion_matrix_ultralytics.png")


# ═════════════════════════════════════════════════════════════════════════════
# 7. SAMPLE PREDICTION IMAGES
# ═════════════════════════════════════════════════════════════════════════════
def save_sample_predictions(class_names, n=16):
    """Run inference on n random val images and save annotated grid."""
    try:
        from ultralytics import YOLO
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return

    val_imgs = sorted((Path(CFG["dataset_root"]) / "images" / "val").glob("*.*"))
    if not val_imgs:
        return

    random.seed(CFG["seed"])
    samples = random.sample(val_imgs, min(n, len(val_imgs)))

    best_pt = OUT / "weights" / "best.pt"
    model   = YOLO(str(best_pt))

    nc      = len(class_names)
    cmap    = plt.cm.get_cmap("tab20", nc)
    colors  = [(int(r*255), int(g*255), int(b*255))
               for r, g, b, _ in [cmap(i) for i in range(nc)]]

    cols    = 4
    rows    = (len(samples) + cols - 1) // cols
    cell_w, cell_h = 320, 240
    grid    = Image.new("RGB", (cols * cell_w, rows * cell_h), (30, 30, 30))

    print(f"\n  Generating {len(samples)} sample predictions …")
    for idx, img_path in enumerate(tqdm(samples, unit="img")):
        results = model.predict(str(img_path), conf=CFG["conf_thresh"],
                                iou=CFG["iou_thresh"], verbose=False)
        img     = Image.open(img_path).convert("RGB")
        draw    = ImageDraw.Draw(img)

        if results[0].boxes is not None:
            for box, cls, conf in zip(
                results[0].boxes.xyxy.cpu().numpy(),
                results[0].boxes.cls.cpu().numpy().astype(int),
                results[0].boxes.conf.cpu().numpy(),
            ):
                x1, y1, x2, y2 = map(int, box)
                color = colors[cls % len(colors)]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                label = f"{class_names[cls]} {conf:.2f}"
                draw.rectangle([x1, y1 - 16, x1 + len(label)*7, y1], fill=color)
                draw.text((x1 + 2, y1 - 14), label, fill=(255, 255, 255))

        img.thumbnail((cell_w, cell_h))
        paste_x = (idx % cols) * cell_w
        paste_y = (idx // cols) * cell_h
        grid.paste(img, (paste_x, paste_y))

    save_path = OUT / "sample_predictions.png"
    grid.save(save_path)
    if SHOW_PLOTS:
        try:
            grid.show()
        except Exception as exc:
            print(f"[WARN] Unable to open image viewer ({exc}).")
    print(f"  ✔ Sample predictions saved → {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# 8. CLASS DISTRIBUTION BAR CHART
# ═════════════════════════════════════════════════════════════════════════════
def plot_class_distribution(class_names):
    """Count label occurrences across train + val and plot a bar chart."""
    root  = Path(CFG["dataset_root"])
    counts = {name: 0 for name in class_names}

    for split in ["train", "val", "test"]:
        lbl_dir = root / "labels" / split
        if not lbl_dir.exists():
            continue
        for lbl_file in lbl_dir.glob("*.txt"):
            with open(lbl_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        if cls_id < len(class_names):
                            counts[class_names[cls_id]] += 1

    if sum(counts.values()) == 0:
        print("[WARN] No labels found — skipping class distribution plot.")
        return

    names  = list(counts.keys())
    values = list(counts.values())
    colors = plt.cm.tab20.colors

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.9), 5))
    bars = ax.bar(names, values, color=colors[:len(names)], edgecolor="white",
                  linewidth=0.7)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_title("Class Distribution (all splits)", fontsize=14)
    ax.set_xlabel("Class")
    ax.set_ylabel("Instance Count")
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "class_distribution.png", dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    print(f"  ✔ Class distribution saved → {OUT}/class_distribution.png")


# ═════════════════════════════════════════════════════════════════════════════
# 9. FINAL SUMMARY TABLE
# ═════════════════════════════════════════════════════════════════════════════
def print_summary(metrics, class_names):
    print("\n" + "="*60)
    print("  FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"  Model weights : {OUT}/weights/best.pt")
    print(f"  mAP@50        : {metrics.box.map50:.4f}")
    print(f"  mAP@50-95     : {metrics.box.map:.4f}")
    print(f"  Precision     : {metrics.box.mp:.4f}")
    print(f"  Recall        : {metrics.box.mr:.4f}")

    if hasattr(metrics.box, "maps"):
        print("\n  Per-class mAP@50:")
        for name, ap in zip(class_names, metrics.box.maps):
            print(f"    {name:<25s} {ap:.4f}")

    print("\n  Saved artefacts:")
    for f in sorted(OUT.glob("*.png")):
        print(f"    {f}")
    print("="*60)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    class_names, nc = check_dataset()

    # ── Class distribution (no training needed) ───────────────────────────────
    print("\n[1/6] Class distribution …")
    plot_class_distribution(class_names)

    # ── Train ─────────────────────────────────────────────────────────────────
    if not EVAL_ONLY:
        print("\n[2/6] Training …")
        train()
    else:
        print("\n[2/6] Skipping training (--eval_only flag set).")

    # ── Validate ──────────────────────────────────────────────────────────────
    print("\n[3/6] Validating …")
    _, metrics = validate(class_names)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    if CFG["skip_extras"]:
        print("\n[4/6] Skipping extra plots/predictions (--skip_extras).")
    else:
        # ── Confusion matrix ──────────────────────────────────────────────────────
        print("\n[4/6] Building confusion matrix …")
        build_confusion_matrix(class_names)

        # ── Training curves ───────────────────────────────────────────────────────
        print("\n[5/6] Plotting training curves …")
        plot_training_curves()
        plot_pr_curve(class_names)

        # ── Sample predictions ────────────────────────────────────────────────────
        print("\n[6/6] Saving sample predictions …")
        save_sample_predictions(class_names, n=16)

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary(metrics, class_names)