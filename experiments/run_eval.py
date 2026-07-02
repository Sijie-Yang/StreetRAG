"""Experiment scripts for npj Urban Sustainability evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


def load_queries(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("queries", data)


def run_baseline_eval(args: argparse.Namespace) -> None:
    """Retrieval baseline comparison (token vs embedding) with P@K/NDCG
    when ground truth annotations are available."""
    from streetrag.core.feature_catalog import FeatureCatalog
    from streetrag.llm.client import load_rag_settings
    from streetrag.llm.retrieval import list_features_info, rank_features_for_query

    from metrics import aggregate, evaluate_ranking

    catalog = FeatureCatalog(args.registry)
    settings = load_rag_settings(str(catalog.path))
    registry_path = str(catalog.path)
    features = list_features_info(catalog)
    queries = load_queries(Path(args.queries))

    gt_path = Path(args.ground_truth)
    ground_truth: dict = {}
    if gt_path.exists():
        with open(gt_path, "r", encoding="utf-8") as f:
            ground_truth = {
                k: v for k, v in json.load(f).items() if not k.startswith("_")
            }

    results = []
    per_method_metrics: dict = {"token": [], "embedding": []}
    for q in queries:
        text = q.get("query", q.get("text", ""))
        qid = q.get("id")
        for method in ("token", "embedding", "stratified"):
            ranked = rank_features_for_query(
                text, features, top_m=args.top_m,
                registry_path=registry_path,
                settings=settings,
                method=method,
            )
            ranked_names = [f["name"] for f in ranked]
            row = {
                "query_id": qid,
                "query": text,
                "method": method,
                "top_features": ranked_names[:10],
            }
            gt = ground_truth.get(qid, {})
            relevant = gt.get("relevant_features") or []
            if relevant:
                row["metrics"] = evaluate_ranking(ranked_names, relevant)
                per_method_metrics[method].append(row["metrics"])
            results.append(row)

    summary = {
        method: aggregate(rows)
        for method, rows in per_method_metrics.items()
        if rows
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)
    if summary:
        for method, m in summary.items():
            print(f"{method}: " + ", ".join(f"{k}={v:.3f}" for k, v in m.items()))
    else:
        print("No ground truth annotations found — wrote rankings only.")
    print(f"Wrote {len(results)} results to {out}")


def run_reproducibility(args: argparse.Namespace) -> None:
    """Run same query N times and compare IndexPlan stability."""
    from streetrag.core.feature_catalog import FeatureCatalog
    from streetrag.llm.client import load_rag_settings
    from streetrag.llm.retrieval import list_features_info, plan_index, rank_features_for_query

    catalog = FeatureCatalog(args.registry)
    settings = load_rag_settings(str(catalog.path))
    registry = catalog.to_legacy_registry()
    features_all = list_features_info(catalog)
    existing = catalog.list_indices()

    runs = []
    for i in range(args.n_runs):
        features = rank_features_for_query(
            args.query, features_all, 60,
            registry_path=str(catalog.path),
            settings=settings,
        )
        plan = plan_index(
            user_query=args.query,
            settings=settings,
            registry_path=str(catalog.path),
            registry=registry,
            features_info=features,
            existing_indices=existing,
        )
        runs.append({
            "run": i,
            "index_name": plan.index_name,
            "operator": plan.operator,
            "features": {fw.name: fw.weight for fw in plan.features},
        })

    out = Path(args.output)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"query": args.query, "runs": runs}, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.n_runs} runs to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="StreetRAG experiment runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_base = sub.add_parser("baseline", help="Retrieval baseline comparison")
    p_base.add_argument("--registry", default="data/feature_registry.json")
    p_base.add_argument("--queries", default="experiments/queries.json")
    p_base.add_argument("--ground-truth", default="experiments/ground_truth.json")
    p_base.add_argument("--top-m", type=int, default=20)
    p_base.add_argument("--output", default="experiments/results/baseline.json")
    p_base.set_defaults(func=run_baseline_eval)

    p_rep = sub.add_parser("reproducibility", help="Plan stability across N runs")
    p_rep.add_argument("query")
    p_rep.add_argument("--registry", default="data/feature_registry.json")
    p_rep.add_argument("--n-runs", type=int, default=5)
    p_rep.add_argument("--output", default="experiments/results/reproducibility.json")
    p_rep.set_defaults(func=run_reproducibility)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
