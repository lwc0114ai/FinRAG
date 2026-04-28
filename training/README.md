# 微调（SFT + LoRA / QLoRA）占位说明

- `configs/sft_lora.example.yaml`：超参与路径示例。  
- `scripts/train_sft.py`：依赖 `.[train]` 安装 `torch` / `transformers` / `peft` / `trl` 后使用；在 CPU 上仅作干跑/冒烟，不保证收敛。

**数据约定**：`training/data/*.jsonl`，每行一个对象，需至少包含可映射到 SFT 的 `instruction` / `input` / `output` 字段，详见 `../docs/03-项目开发文档.md`。

**合规提示**：请确保训练数据已获授权并符合内部合规与脱敏要求。
