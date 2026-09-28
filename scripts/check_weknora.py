"""WeKnora 检索后端自检脚本

用法：
  .venv/bin/python scripts/check_weknora.py                 # 用默认问题探一条
  .venv/bin/python scripts/check_weknora.py "小甸集特支"     # 指定问题

做的事：读 config.yaml -> 构造 WeKnoraRetriever -> 打一次 hybrid-search，
并打印转换后的 hits（与 core/retriever.Retriever 同形状），用于确认项目与
WeKnora 的字段映射、连通性以及分值区间是否符合预期。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import load_config
from core.weknora import WeKnoraRetriever, WeKnoraUnavailable

cfg = load_config()


def main() -> int:
    query = sys.argv[1] if len(sys.argv) > 1 else "中共小甸集特支为什么重要？"
    if (cfg.get("retrieval") or {}).get("backend", "local") != "weknora":
        print("提示：config.yaml 里 retrieval.backend 当前不是 weknora，本脚本仍按 weknora 配置探测。\n")

    r = WeKnoraRetriever(cfg)
    info = r.describe()
    print("配置：")
    print(f"  base_url          = {info['base_url']}")
    print(f"  knowledge_base_id = {info['knowledge_base_id'] or '(未配置)'}")
    print(f"  api_key           = {'已配置' if info['api_key_configured'] else '未配置'}")
    print(f"\n探测问题：{query}\n")

    try:
        hits = r.probe(query)
    except WeKnoraUnavailable as e:
        print(f"[失败] {e}")
        return 1

    if not hits:
        print("[空] 检索没有返回任何分块：确认该知识库已上传文档并完成解析与向量化。")
        return 1

    for i, h in enumerate(hits, 1):
        print(f"[{i}] score={h['score']:.4f}  出处：《{h['doc']}》  chunk={h['chunk_id']}")
        print(f"    {h['text'][:160]}…")
    print("\n字段映射：content->text / knowledge_title->doc,title / score->score,dense_score ✓")
    print("若 score 区间与本地余弦分差异较大，请相应调整 config.yaml 的")
    print("weknora.refuse_threshold 与 thresholds.* 后再切换 retrieval.backend。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
