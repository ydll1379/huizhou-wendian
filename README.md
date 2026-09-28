# 徽州问典

**徽州红色文化智能传承与乡村文旅赋能平台**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/数据-公开资料整理-blue)](#数据说明)

> 让每一次提问都有出处，让每一次行走都有收获。

通用大模型讲红色历史容易"一本正经地胡说"，传统景区导览又无法追问。**徽州问典**用 RAG（检索增强生成）技术构建徽州红色文化专属知识库，让 AI 讲解红色文化**有据可查、有源可溯**；并把讲解延伸到线下——生成研学路线、记录打卡、产出实践报告，服务"青年红色筑梦之旅"与乡村振兴。

## 目录

- [核心功能](#核心功能)
- [效果示例](#效果示例)
- [技术架构](#技术架构)
- [快速开始](#快速开始)
- [可选：用 WeKnora 作检索后端](#可选用-weknora-作检索后端)
- [项目结构](#项目结构)
- [优化实验](#优化实验)
- [数据说明](#数据说明)
- [路线图](#路线图)

## 核心功能

### 1. 红色文化智能问答（带引用溯源）

提问后不仅给出讲解，还标注每一句回答的原始文献出处，可点击查看原文片段。

- **三类人群版本**：研学教师（课堂讲解与研学任务）/ 红色文化游客（现场易懂的故事讲解）/ 基层文旅宣传助手（可复制修改的景点介绍、活动预告与传播文案）
- **场景快捷入口**：按当前人群展示可直接发起的备课、游览或宣传任务，回答可一键复制
- **教师备课参数**：设置学生学段、讲解时长和教学目标，输出课堂提纲、互动问题与现场任务
- **宣传素材模板**：选择景点简介、公众号推文、活动预告、短视频口播、讲解词或海报文案；缺失的活动信息会标为待补充
- **低置信度拒答**：检索相关度低于阈值时直接拒答，而不是编造——红色主题对事实准确性要求极高
- **问法自适应**：问"包青天"也能命中写"包拯"的文献（领域词表扩展）

### 2. 红色筑梦之旅 · 研学路线平台

面向青年实践的完整闭环：

| 环节 | 说明 |
|---|---|
| 路线规划 | 可指定起点、终点与必经景点；大模型编排入选理由和研学任务，高德路径规划计算站间交通（步行/驾车、距离、耗时） |
| 任务打卡 | 每站完成研学任务后记录现场心得，实时统计进度 |
| 实践报告 | 一键生成庄重规范的 Markdown 实践报告：誊录润色心得、统计完成度、撰写实践感悟，可复制/下载 |

### 3. LBS 融合推荐（文化 + 位置）

提问提到景点时，自动出现继续安排探访的入口；用户按需加载周边餐饮或住宿，也可以跳转到该城市生成研学路线：

- 高德地图真实 POI：美食 6 条 / 住宿 4 条，按距离排序，附导航链接与静态地图
- **每条推荐附文化溯源**：以菜品、地标命名的店铺会命中知识库原文并标注出处
- 26 处景点注册表覆盖合肥、芜湖、阜阳、淮南、六安、滁州、宣城

### 4. 轻量反馈与原型说明

- 可提交史料纠错线索或功能建议，反馈保存在本机 `data/feedback.json`，不收集姓名、电话等个人信息
- 页面说明项目为公益演示原型，AI 内容仅供辅助参考，不能替代场馆专业讲解；地图 POI 和导航由高德地图提供
- 本机维护页可对反馈标记待处理、已核实、已处理或不采纳，并记录处理备注；审核接口拒绝非本机请求
- 知识片段会保留原始文件路径，并读取 front matter 中明确填写的来源链接、发布机构、日期、适用地点和页码；给已有索引补路径字段可运行 `.venv/bin/python scripts/enrich_kb_metadata.py`，新增 front matter 后需重跑建库脚本。当前原始资料尚未提供可识别的来源 URL、发布机构和页码，补齐前引用卡只显示文件路径

## 效果示例

**红色历史问答**（含出处标注，`[n]` 可点击查看原文）：

> **问**：中共小甸集特支为什么重要？
>
> **答**：中共小甸集特支于 1923 年冬在寿县小甸集成立，是安徽省成立最早的中共农村党组织，直属党中央领导 [1][2][5]。它点燃了安徽革命斗争的火种，开启了安徽人民革命斗争史的新纪元 [1]……
>
> 📚 出处：《安徽第一面党旗纪念园》《寿县小甸村》

**研学路线**（真实交通数据）：

> 🚩 淮南红色一日研学线
> 安徽第一面党旗纪念园 → 瓦埠暴动旧址 → 淮南新四军纪念林景区 → 侵华日军淮南罪证遗址 → 三里沟抗日纪念园 → 板张集革命烈士陵园
> （站间驾车 22~56 公里，由高德路径规划 API 实时计算）

**周边推荐**：

> 三河古镇周边美食：韦家私房菜（40 米）、百年中和祥（52 米，📖 附《三河古镇》文化溯源）、老彭米饺店（69 米）……

## 技术架构

```
                          文档层
   红色文献 / 地方志 / 纪念馆资料 / 非遗名录 / 扫描版图书
                             │  解析：PDF / HTML / Markdown / macOS Vision OCR（中文）
                             ▼
                          知识层
   标题感知分块 → BGE-m3 中文向量化 → 混合检索（BM25 + 语义向量，RRF 融合）
                             │  领域词表扩展（庐州=合肥、包青天=包拯、芍陂=安丰塘…）
                             ▼
                          排序层
   BGE-Reranker-v2-m3 精排 → 相关度阈值判定（低于阈值拒答）
                             │
                             ▼
                          生成层
   DeepSeek API → 人群适配 Prompt（研学教师/游客讲解/宣传助手）→ 强制引用标注 → 溯源输出
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
          应用层：Web 问答界面        高德 LBS：周边推荐 / 路线规划 / 实践报告
```

### 技术选型

| 层 | 选择 | 说明 |
|---|---|---|
| 文档解析 | pypdf / BeautifulSoup / OpenCC | 扫描版用 macOS Vision 框架 OCR（148 页中文图书 77 秒完成） |
| 分块 | 标题感知 + 句子边界滑动窗口 | 保留标题上下文，避免切断语义 |
| Embedding | BAAI/bge-m3（本地运行） | 中文最优开源向量模型之一，无需 API |
| 向量库 | 自研 numpy 余弦检索 | 数千分块规模下毫秒级，全链路可控便于做对比实验 |
| 检索 | BM25 + 向量混合（RRF 融合） | 关键词与语义互补，中文专有名词场景提升明显 |
| 重排序 | BAAI/bge-reranker-v2-m3 | 精排提升首位命中率 |
| 生成 | DeepSeek API（OpenAI 兼容） | 中文强、成本低，可替换为任意兼容服务 |
| 地图 | 高德地图 Web 服务 API | POI 搜索、周边搜索、路径规划、静态地图 |
| 后端 | FastAPI | 轻量、异步、自带 OpenAPI 文档 |
| 前端 | 原生 HTML/CSS/JS | 零构建步骤，单文件即可部署 |

**为什么不用 RAGFlow 等重型框架？** 本项目所有优化项（词表扩展、分块策略、混合检索、重排、拒答）都需要对检索链路有完全控制权才能做 A/B 对比实验；轻量自研方案 2000 行代码即可跑通，且无需 Docker、易于二次开发和部署演示。

## 快速开始

前置要求：Python 3.9+，无需 Docker。（首次运行会经 hf-mirror 自动下载 bge-m3 模型约 2GB）

```bash
# 1. 克隆并安装依赖
git clone https://github.com/ydll1379/huizhou-wendian.git
cd huizhou-wendian
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. 配置 API key
cp .env.example .env
#   DEEPSEEK_API_KEY —— platform.deepseek.com 申请（问答生成，必填）
#   AMAP_API_KEY     —— lbs.amap.com 申请「Web服务」类型（周边推荐/路线，选填）

# 3. 建库（把 data/raw/ 下的文档向量化）
.venv/bin/python scripts/build_kb.py

# 4. 启动服务
.venv/bin/uvicorn app.main:app --port 8000

# 5. 浏览器打开 http://127.0.0.1:8000
```

> 让局域网内其他电脑访问：启动命令加 `--host 0.0.0.0`
>
> 前端右上角切换 **💬 文化问答** / **🚩 红色筑梦**；本机运行者也可打开反馈审核页。首期重点试点为合肥、芜湖、阜阳、淮南，注册表当前覆盖其他安徽城市的部分景点，具体以界面可选范围为准。

## 可选：用 WeKnora 作检索后端

检索层支持整体替换：把 [Tencent/WeKnora](https://github.com/Tencent/WeKnora)（腾讯开源知识平台）当作召回端，**生成仍由本项目自己的 DeepSeek Prompt 负责**——三类人群版本、`[n]` 引用编号、低置信度拒答、LBS 推荐与筑梦之旅全部不变，换掉的只有检索这一环。

```bash
# 1. 部署 WeKnora（默认监听 http://localhost:8080，需 Docker Desktop 已启动）
git clone https://github.com/Tencent/WeKnora && cd WeKnora && docker compose up -d
#    若拉镜像报错，见下方「部署前提」里的镜像加速说明
#    启动后在 WeKnora 页面创建知识库、上传 data/raw 下的资料，记下知识库 ID（kb-xxxxxxxx）

# 2. 配置本项目
#    .env          填 WEKNORA_BASE_URL / WEKNORA_API_KEY
#    config.yaml   填 weknora.knowledge_base_id，并把 retrieval.backend 改为 weknora

# 3. 自检（打印字段映射与分值区间，便于校准阈值）
.venv/bin/python scripts/check_weknora.py "中共小甸集特支为什么重要？"

# 4. 正常启动，前端与接口完全不变
.venv/bin/uvicorn app.main:app --port 8000
```

适配器在 [core/weknora.py](core/weknora.py)，只调用 `POST /api/v1/knowledge-bases/{id}/hybrid-search`（请求 `{"query_text", "match_count"}`），把返回的 `content` / `knowledge_title` / `score` 映射成本项目统一的 hits 形状，因此 `pipeline.py`、`generator.py`、`recommender.py`、`route_planner.py` 与前端都无需改动。

### 部署前提

WeKnora 需要自己配一套「对话模型 + 向量模型」，两者角色不同，缺一不可：

| 模型 | 本项目用的 | 说明 |
|---|---|---|
| 对话（KnowledgeQA） | DeepSeek API | 直接复用本项目的 `DEEPSEEK_API_KEY` |
| 向量（Embedding） | Ollama + `bge-m3` | **DeepSeek 不提供向量模型服务**，需本机跑一个 Ollama（`brew install ollama && ollama pull bge-m3`，约 1.2 GB）；WeKnora 通过 `OLLAMA_BASE_URL=http://host.docker.internal:11434` 访问。选 `bge-m3` 是因为它正是本项目本地后端在用的模型，便于公平对比 |

Docker Hub 在国内直连常不通（表现为 `docker pull` 长时间无响应或 000）。给 Docker Desktop 配上国内镜像加速即可，编辑 `~/.docker/daemon.json` 后重启 Docker Desktop：

```json
{ "registry-mirrors": ["https://docker.m.daocloud.io", "https://docker.1panel.live"] }
```

### 实测数据

| 验证项 | 结果 |
|---|---|
| 适配器端到端测试 | 18/18 通过（`.venv/bin/python scripts/test_weknora_adapter.py`，内置模拟服务，**无需真机即可跑**） |
| 本地后端回归 | 改造前后 50 题 Hit@5 / Hit@10 / MRR 完全一致；逐条比对 50 题 × 12 条召回分块，**零差异** |
| 切换后端后的启动开销 | 13.8 秒 / 1.48 GB → **0.0 秒 / 362 MB**（不再加载本地 bge-m3 与 bge-reranker） |

### 三点提醒

- **分值口径会变。** WeKnora 返回的是 rerank 归一化分，与本地 bge-m3 余弦分不同，切换后需用自检脚本重新校准 `config.yaml` 里的 `weknora.refuse_threshold` 与 `thresholds.*`（周边推荐/路线背景的最低相关度）。
- **对比实验仍用本地后端。** `retrieval.baseline` 的消融实验依赖对检索链路的完全控制，跑 `scripts/run_eval.py` 时保持 `backend: local`。
- **官方文档有个坑。** `docs/api/knowledge-search.md` 描述的 `POST /api/v1/knowledge-search` 在当前 main 分支上实际是**文档列表关键词搜索**（`handler.SearchKnowledge` 读的是 URL 上的 `keyword/query` 参数，返回 `has_more/total`），不是向量检索；可用的检索入口是 `POST /api/v1/knowledge-bases/{id}/hybrid-search`。适配器已按源码实现对后者。

## 项目结构

```
huizhou-wendian/
├── config.yaml              # 全局配置：模型、检索参数、领域词表、风格 Prompt、景点注册表
├── core/                    # RAG 核心链路
│   ├── loader.py            #   文档加载（txt/md/html/pdf/docx）
│   ├── chunker.py           #   标题感知 + 滑动窗口分块
│   ├── embeddings.py        #   bge-m3 向量化
│   ├── vector_store.py      #   轻量向量库（numpy 余弦 + 持久化）
│   ├── retriever.py         #   混合检索 + 领域词表扩展（含 baseline 开关）
│   ├── reranker.py          #   BGE-Reranker 精排
│   ├── weknora.py           #   可选：WeKnora 检索后端适配器（backend=weknora）
│   ├── generator.py         #   DeepSeek 生成（引用标注 + 庄重约束 + 拒答）
│   ├── pipeline.py          #   RAG 总管线
│   ├── amap.py              #   高德地图客户端（QPS 节流 + 重试）
│   ├── recommender.py       #   景点识别 + 周边推荐 + 文化溯源
│   ├── route_planner.py     #   研学路线 / 打卡 / 实践报告
│   └── llm.py               #   LLM 调用助手（JSON / 文本）
├── app/
│   ├── main.py              #   FastAPI 接口
│   └── static/index.html    #   Web 界面（问答页 + 筑梦页）
├── scripts/
│   ├── build_kb.py          #   建库脚本（新增资料后重跑即可）
│   ├── run_eval.py          #   检索优化对比实验
│   ├── check_weknora.py     #   WeKnora 检索后端连通性自检
│   ├── test_weknora_adapter.py # 适配器端到端测试（模拟 WeKnora 服务，无需真机）
│   └── fetch_data.py        #   公开网页资料采集辅助
├── eval/
│   ├── questions.json       #   50 道红色主题评测题
│   └── results/             #   实验报告（版本管理）
└── data/
    ├── raw/                 #   知识库原始文档（55 份，按主题分目录）
    │   ├── 红色/            #     红色景点、红色村庄、革命老区
    │   ├── 合肥/ 淮南/ 芜湖/ 阜阳/   # 地方文史
    │   ├── 淮南特产/        #     服务乡村文旅赋能
    │   └── seed/            #     非遗等补充资料
    └── kb/                  #   生成的向量库（.gitignore，clone 后重建）
```

## 优化实验

答辩硬通货：检索优化的前后对比数据可一键复现。

```bash
.venv/bin/python scripts/run_eval.py          # 全量 50 题
.venv/bin/python scripts/run_eval.py --quick  # 快速 5 题
```

最新结果（50 道红色主题题，[完整报告](eval/results/)）：

| 指标 | 基线（纯向量） | 优化（词表扩展+混合检索+Rerank） | 提升 |
|---|---|---|---|
| Hit@5 | 100.0% | 100.0% | +0.0% |
| Hit@10 | 100.0% | 100.0% | +0.0% |
| MRR | 0.953 | 0.967 | **+0.013** |

> 说明：当前知识库规模下 Hit@k 已饱和，MRR（平均倒数排名）体现的是 **Reranker 对排序质量的增益**——正确结果被排到更靠前的位置，直接影响生成质量。随着知识库扩充，Hit 指标差距将进一步拉开。

## 数据说明

知识库 55 份文档（184 个分块），来源均为**公开资料整理**，每份文档标注来源：

| 主题 | 内容 | 来源示例 |
|---|---|---|
| 淮南红色（23 份） | 16 处红色景点、3 个红色村庄、村史馆集锦 | 《淮南红色旅游资源概览》（市委党史和地方志研究室编纂，黄山书社 2023）OCR |
| 革命老区（3 份） | 金寨红军广场、小岗村、云岭新四军军部 | 六安市政府、泾县政府官网、人民网、新华网等 |
| 地方文史（24 份） | 合肥/淮南/芜湖/阜阳 文保单位与景区 | 市县政府官网、博物馆公开资料 |
| 非遗补充（5 份） | 包公故事、庐剧、粉蜡笺、巢湖民歌等 | 公开出版物与官方网站整理 |

⚠️ 本项目为学习与竞赛用途，数据仅供研究演示，请遵守各来源网站的版权规定。正式使用时请核实并补充一手文献。

## 路线图

- [x] RAG 问答 + 引用溯源 + 多风格讲解
- [x] 高德 LBS 融合推荐（周边美食/住宿 + 文化溯源）
- [x] 红色筑梦之旅路线平台（规划 + 打卡 + 实践报告）
- [x] 检索优化对比实验框架（50 题评测 + MRR）
- [ ] 语音提问与讲解播报（Web Speech API）
- [ ] 知识库扩充至 150+ 份，覆盖更多革命老区
- [ ] 多租户"一城一库"复制模式

## 致谢

- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) · [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) —— 开源中英文向量与重排序模型
- [DeepSeek](https://platform.deepseek.com/) —— 大模型 API
- [高德开放平台](https://lbs.amap.com/) —— 地图与位置服务

---

本仓库为"青年红色筑梦之旅"赛道参赛项目《徽州问典》工程实现。
