"""
mlops/save_demo_frames.py
Copy the 40 demo frames from your Colab sequence into data/demo_frames/
so every MLflow run uses identical frames for fair comparison.

Usage (Colab):
    !python mlops/save_demo_frames.py \
        --seq_dir /content/visdrone_raw/uav0000288_00001_v \
        --max_frames 40

Usage (local, point to any folder of images):
    python mlops/save_demo_frames.py --seq_dir /path/to/frames
"""
import argparse, shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_dir",    type=Path, required=True)
    parser.add_argument("--out_dir",    type=Path, default=Path("data/demo_frames"))
    parser.add_argument("--max_frames", type=int,  default=40)
    args = parser.parse_args()

    imgs = sorted(list(args.seq_dir.glob("*.jpg")) +
                  list(args.seq_dir.glob("*.png")))[:args.max_frames]
    if not imgs:
        print(f"[ERROR] No images found in {args.seq_dir}")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(imgs):
        dst = args.out_dir / f"frame_{i:04d}{p.suffix}"
        shutil.copy2(p, dst)
    print(f"Saved {len(imgs)} frames → {args.out_dir}")

if __name__ == "__main__":
    main()
