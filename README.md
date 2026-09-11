# 徽州问典 —— 江淮四城（合肥·芜湖·阜阳·淮南）地方文化智能讲解与传承平台

针对合肥地方志、非遗资料"分散、难读、没人看"的痛点，基于 RAG（检索增强生成）搭建合肥地方文化专属知识库 + 智能讲解助手。游客、学生、研究者"问一句话，就能听懂庐州故事"，每个回答都能溯源到原始文献。

## 技术栈

| 层 | 选型 |
|---|---|
| 文档解析 | pypdf / BeautifulSoup / OpenCC（繁简转换） |
| 分块 | 标题感知 + 滑动窗口（`core/chunker.py`） |
| Embedding | BAAI/bge-m3（本地运行） |
| 向量库 | 自研 numpy 余弦检索（`core/vector_store.py`） |
| 检索 | BM25 + 向量混合（RRF 融合）+ 领域词表查询扩展 |
| 重排序 | BAAI/bge-reranker-v2-m3 |
| 生成 | DeepSeek API（OpenAI 兼容） |
| 前端 | FastAPI + 原生 HTML/JS 对话界面 |

## 快速开始（clone 后 5 步跑通）

前置要求：Python 3.9+，无需 Docker。

```bash
# 1. 克隆并安装依赖
git clone https://github.com/ydll1379/huizhou-wendian.git
cd huizhou-wendian
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. 配置 API key（复制 .env.example 为 .env，填入两把 key）
cp .env.example .env
#   DEEPSEEK_API_KEY —— platform.deepseek.com 申请（问答生成，必填）
#   AMAP_API_KEY    —— lbs.amap.com 申请「Web服务」类型（周边推荐，选填）

# 3. 建库（首次运行自动经 hf-mirror 下载 bge-m3 模型，约 2GB，请耐心等待）
.venv/bin/python scripts/build_kb.py

# 4. 启动 Web 服务
.venv/bin/uvicorn app.main:app --port 8000

# 5. 浏览器打开 http://127.0.0.1:8000
```

> 让同一局域网的电脑/手机访问：启动命令加 `--host 0.0.0.0`，对方访问 `http://<你的内网IP>:8000`。

## 项目结构

```
luzhou-wendian/
├── config.yaml          # 全局配置（模型、检索参数、领域词表、风格 Prompt）
├── core/                # RAG 核心链路
│   ├── loader.py        #   文档加载（txt/md/html/pdf）
│   ├── chunker.py       #   分块策略
│   ├── embeddings.py    #   bge-m3 向量化
│   ├── vector_store.py  #   轻量向量库（numpy 余弦）
│   ├── retriever.py     #   混合检索 + 词表扩展（baseline 开关）
│   ├── reranker.py      #   BGE-Reranker 重排
│   ├── generator.py     #   DeepSeek 生成（引用标注 + 拒答）
│   └── pipeline.py      #   总管线
├── app/                 # Web 服务（FastAPI + 对话界面）
├── scripts/
│   ├── build_kb.py      #   建库脚本
│   ├── run_eval.py      #   评测脚本（基线 vs 优化对比实验）
│   └── fetch_data.py    #   公开网页数据收集（Phase 1）
├── eval/                # 评测题 + 实验结果
└── data/
    ├── raw/             # 原始文档（放进来的文档即入库）
    └── kb/              # 生成的向量库
```

## 核心功能

1. **文化智能问答**：回答自动标注 `[编号]` 出处，可点击查看原文片段
2. **多风格讲解**：导游版 / 学者版 / 儿童版，Prompt 在 `config.yaml` 可调
3. **引用溯源**：回答附参考文献列表（文档名 + 原文片段 + 相关度）
4. **拒答防幻觉**：召回相关度低于阈值时拒绝回答，而不是编造

## 优化实验（答辩硬通货）

```bash
# 跑基线（纯向量）vs 优化（词表扩展+混合检索+Rerank）对比
.venv/bin/python scripts/run_eval.py            # 全量 12 题
.venv/bin/python scripts/run_eval.py --quick    # 快速 5 题
```

结果输出到 `eval/results/<时间戳>/`：`baseline.json`、`optimized.json`、`report.md`（Hit@5 / Hit@10 对比表）。评测题在 `eval/questions.json`，按团队实际知识库扩充到 50~100 题。

## 周边推荐（高德地图 LBS）

问答之外，可对知识库收录的 25 处景点做「文化+位置」融合推荐：

1. 在 [lbs.amap.com](https://lbs.amap.com) 控制台免费申请 **「Web服务」** 类型 key
2. 填入 `.env` 的 `AMAP_API_KEY`，重启服务

- 提问中提到景点（含别名，如"李府"→李鸿章故居）时，回答下方自动出现**周边推荐面板**
- 美食/住宿推荐每条附**文化溯源**（知识库原文片段+出处）与高德导航链接
- 接口：`POST /api/recommend {spot, kind}`、`GET /api/spots`
- 景点注册表与别名在 `config.yaml` 的 `spots` 段维护

## 下一步（按项目文档的 15 天计划）

- **Phase 1 数据收集**：扩充 `data/raw/` 至 80+ 份文档（非遗名录、包公故事、方志扫描版等），`scripts/fetch_data.py` 是抓取入口，也可手工整理网页另存为 md
- **Phase 2 优化**：调整 `config.yaml` 的领域词表与检索参数；接入 OCR 处理扫描版方志（繁简转换用 OpenCC）
- **Phase 3 加分项**：文旅路线生成、TTS 语音播报、知识图谱可视化（选做）

## 说明

`data/raw/seed/` 下的 8 份示例文档为演示用种子数据（基于公开常识整理），正式知识库请用真实收集的资料替换/扩充。所有数据仅用于学习演示，请遵守来源网站版权规定。
