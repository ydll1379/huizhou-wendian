# 在另一台电脑上部署《徽州问典》

> 本文件面向"换一台机器从零跑起来"的场景。开发机环境：macOS + Python 3.9.6；
> 代码与公开资料在 GitHub，**密钥与向量库不入库**，需要在新机器上自行配置与重建。

---

## 0. 前置条件

| 项目 | 要求 | 说明 |
| --- | --- | --- |
| Python | 3.9 及以上 | 开发机为 3.9.6，3.10/3.11 亦可 |
| git | 任意版本 | 拉取代码 |
| 磁盘 | ≥ 8 GB | 依赖约 2 GB + 两个模型缓存约 5 GB |
| 内存 | ≥ 8 GB | 本地向量与重排模型在 CPU 上运行 |
| 网络 | 可访问 api.deepseek.com | 生成回答必填；知识库检索与建库可离线 |
| Docker | 可选 | 仅当使用 WeKnora 作检索后端时需要 |

---

## 1. 拉取代码

```bash
git clone https://github.com/ydll1379/huizhou-wendian.git
cd huizhou-wendian
```

## 2. 创建虚拟环境并安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

首次安装会拉取 PyTorch（CPU 版）与 sentence-transformers，约 2 GB，耗时较长。
若所在网络访问 PyPI 慢，可加国内镜像：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 3. 配置密钥（必须）

`.env` 不入库，需要新建：

```bash
cp .env.example .env
```

编辑 `.env`：

```ini
DEEPSEEK_API_KEY=sk-你的key          # 必填，platform.deepseek.com 申请
AMAP_API_KEY=你的高德Web服务key       # 选填，lbs.amap.com 申请「Web服务」类型
```

程序启动时会自动读取 `.env`（`core/__init__.py` 中的 `load_dotenv`），无需手动 export。
若同时设置了系统环境变量，优先级高于 `.env`。

## 4. 重建知识库（必须）

`data/kb/`（向量与分块）**不入库**，但**原始资料已入库**（`data/raw/`，含合肥、淮南、芜湖、阜阳与"江淮地区增补"等）。新机器上执行：

```bash
python scripts/build_kb.py
```

该脚本会扫描 `data/raw/` 下全部文档 → 分块 → 用 `BAAI/bge-m3` 生成向量 → 写入 `data/kb/`。

**模型下载**：首次运行会下载 `BAAI/bge-m3`（向量）与 `BAAI/bge-reranker-v2-m3`（重排），
各约 2 GB。国内网络建议先设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com     # Windows PowerShell: $env:HF_ENDPOINT="https://hf-mirror.com"
python scripts/build_kb.py
```

模型下载完成后，之后离线运行可加：

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

CPU 建库约几分钟到十几分钟；开发机实测 391 个分块、向量维度 1024。

## 5. 启动服务

```bash
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

浏览器打开 <http://127.0.0.1:8000>。

局域网内其他设备（手机）访问，把 host 换成 `0.0.0.0`：

```bash
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
# 手机访问 http://<这台电脑的局域网IP>:8000
```

## 6. 自检

```bash
curl http://127.0.0.1:8000/api/health
```

正常应返回：

```json
{"status":"ok","retrieval_backend":"local","kb_ready":true,"styles":["teacher","tourist","promo"]}
```

`kb_ready` 为 `false` 说明第 4 步没跑成功；页面能打开但提问报"未找到 DEEPSEEK_API_KEY"说明第 3 步没配好。

---

## 7. 可选：用 WeKnora 作检索后端

默认使用自研本地检索（`retrieval.backend: local`）。若要切换：

```bash
git clone https://github.com/Tencent/WeKnora && cd WeKnora && docker compose up -d
```

在 `.env` 中补 `WEKNORA_BASE_URL`、`WEKNORA_API_KEY`，并把 `config.yaml` 里的
`retrieval.backend` 改为 `weknora`（该分支拒答阈值口径不同，见配置注释，需实测校准）。
另有对照脚本 `scripts/compare_backends.py`、`scripts/check_weknora.py` 可用。

## 8. 常见问题

| 现象 | 原因与处理 |
| --- | --- |
| 启动报 `未找到 DEEPSEEK_API_KEY` | `.env` 未创建或键名写错；确认在仓库根目录且文件名是 `.env` |
| 提问后长时间无响应 | 首次调用需加载模型；CPU 上重排较慢（实测 P50 约 5 s） |
| 建库卡在下载模型 | 设置 `HF_ENDPOINT=https://hf-mirror.com` 后重试；或先手动下载模型到 HF 缓存 |
| 端口被占用 | `--port 8001` 换端口；或结束旧进程 `pkill -f "uvicorn app.main:app"` |
| 页面能开、搜索无结果 | `data/kb/` 为空，重跑第 4 步 |
| 周边推荐/路线无数据 | 未配置 `AMAP_API_KEY`；地图相关功能会降级但不影响问答 |

## 9. 不进版本库的内容

- `.env`（密钥）
- `.venv/`（虚拟环境）
- `data/kb/`（向量与分块，用 `scripts/build_kb.py` 重建）
- 运行时用户数据：`data/routes.json`、`data/checkins.json`、`data/reports/`、`data/feedback.json`
- `.drawio-tmp/`（画图临时目录）

## 10. 目录速览

```
app/            FastAPI 服务与前端页面（app/static/index.html）
core/           检索、生成、评测、路线、LBS、WeKnora 适配等核心模块
scripts/        建库、评测、对照实验与自检脚本
data/raw/       公开史料原文（入库）
data/kb/        向量库与分块（不入库，需重建）
docs/           需求分析、技术路线与架构设计
eval/           评测集与历史评测产物
presentation/   演示截图与素材
config.yaml     模型、检索、门控、场景提示词等全部配置
requirements.txt
```
