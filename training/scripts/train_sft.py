"""
SFT + LoRA 训练占位脚本。真实训练: pip install -e ".[train]" 并接入 TRL、PEFT。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("training/configs/sft_lora.example.yaml"))
    args = ap.parse_args()
    if not args.config.is_file():
        print(f"Missing config: {args.config}", file=sys.stderr)
        sys.exit(1)
    # 极简键值解析，避免依赖 PyYAML
    cfg: dict[str, str] = {}
    for line in args.config.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        cfg[k.strip()] = v.strip()
    out = {
        "status": "stub",
        "message": "请在此对接 TRL SFTTrainer + PEFT。参考 training/README.md 与 finrag 文档。",
        "resolved": {k: cfg.get(k) for k in ("base_model", "train_file", "output_dir") if k in cfg},
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    tfile = Path(cfg.get("train_file", "training/data/train.jsonl"))
    if not tfile.is_file():
        print(f"提示: 训练数据尚未就绪: {tfile}")


if __name__ == "__main__":
    main()
