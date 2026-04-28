# 金融 RAG + 微调（MVP 可运行项目）

基于文档中的目标实现：**FastAPI** 服务、**LangChain** 编排、**Chroma** 本地向量库、**SQLite** 元数据与任务、**OpenAI 兼容 API** 调用大模型与 Embedding；并附带 **LoRA/QLoRA 训练脚本占位** 与 `docker-compose` 示例。

## 环境

- Python 3.10+（建议 3.11），推荐使用项目文档中的 Conda 环境，例如已存在的 `ai0114_rag`（路径示例：`E:\miniconda3\envs\ai0114_rag`）。

```powershell
# 激活 Conda 环境（请按你本机 miniconda/anaconda 路径调整）
E:\miniconda3\Scripts\activate
conda activate ai0114_rag

cd "c:\Users\0\Desktop\深度学习练习\project1"
pip install -e ".[dev]"
```

## 配置

1. 复制 `\.env.example` 为 `\.env`。
2. 生产/真实调用时填写 `OPENAI_API_KEY` 与（可选）`OPENAI_BASE_URL`（如硅基流动、通义、DeepSeek 等 OpenAI 兼容端点）及 `LLM_MODEL` / `EMBEDDING_MODEL`。
3. 本地自动化测试/无网环境可设 `DEV_MOCK_LLM=true`（将使用假 Embedding 与 `FakeListChatModel`，不访问外网）。

## 启动

```powershell
# 在已激活的 ai0114_rag 环境中
cd "c:\Users\0\Desktop\深度学习练习\project1"
uvicorn finrag.api.main:app --reload --host 0.0.0.0 --port 8000
```

浏览器打开 <http://127.0.0.1:8000/docs> 进行调试。

等效命令：`python -m finrag`（默认 8000 端口并开启 reload）。

## 主要 API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/v1/health` | 健康检查 |
| POST | `/v1/knowledge_bases` | 创建知识库 |
| GET | `/v1/knowledge_bases` | 列表 |
| POST | `/v1/knowledge_bases/{kb_id}/documents` | 上传 txt/md/pdf，异步索引，返回 `job_id` |
| GET | `/v1/jobs/{job_id}` | 查询索引进度 |
| POST | `/v1/chat` | 非流式 RAG 问答，返回 `citations` |
| POST | `/v1/chat/stream` | 流式回答，文末可能带 `__CITATIONS__=...` 行（便于调试） |

若环境变量中设置了 `API_KEY`，则请求头需带 `X-API-Key`。

## 数据目录

- `data/storage/` 上传的原始文件  
- `data/chroma/` Chroma 持久化（勿与真模型的 Embedding 与 mock 混用，切换时请清空或换目录）  
- `data/derived/{kb_id}/all_chunks.jsonl` 为混合检索（BM25）使用的块副本  
- `data/finrag.db` SQLite  

## 测试

```powershell
# 在 ai0114_rag 中
cd "c:\Users\0\Desktop\深度学习练习\project1"
$env:DEV_MOCK_LLM="true"
pytest -q
```

## 项目结构

详见 `docs/03-项目开发文档.md`；代码包名为 `finrag`，位于 `src/finrag/`。

## 训练（可选依赖）

```powershell
pip install -e ".[train]"
python training/scripts/train_sft.py --config training/configs/sft_lora.example.yaml
```

需自备 JSONL 指令数据、基座模型与 GPU；脚本中均为占位与示例路径，请按实环境修改。
