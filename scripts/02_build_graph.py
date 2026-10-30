"""Build the review graph: from the raw text release (rebuilt) or from the public .mat file.
Writes data/processed/graph_<source>[_smoke].npz (+ _nodes.parquet for raw)."""
from __future__ import annotations

import numpy as np
from _common import ROOT, load_config, parser, processed_path, resolve_source

from camo.data.features_pit import FEATURES, compute_pit_features
from camo.data.graph_build import build_graph, compare_to_published
from camo.data.load_mat import load_mat, save_graph
from camo.data.load_raw import load_raw
from camo.eval.runlog import RunLog


def main():
    args = parser(__doc__).parse_args()
    cfg = load_config("base")
    source = resolve_source(args.source)
    log = RunLog(f"02_build_graph_{source}", cfg)

    if source == "mat":
        g = load_mat(ROOT / cfg["paths"]["mat_dir"] / "YelpChi.mat")
        if args.smoke:
            rng = np.random.default_rng(0)
            g = g.subgraph(np.sort(rng.choice(g.n, cfg["smoke"]["n_nodes"], replace=False)))
    else:
        raw_dir = ROOT / cfg["paths"]["raw_dir"]
        have_raw = raw_dir.exists() and any(p.name.startswith("metadata") for p in raw_dir.glob("*"))
        if args.smoke and not have_raw:
            raw_dir = ROOT / "tests" / "fixtures" / "raw_mini"
        raw = load_raw(raw_dir)
        if args.smoke and len(raw) > cfg["smoke"]["n_nodes"]:
            raw = raw.sample(cfg["smoke"]["n_nodes"], random_state=0).sort_values("review_id")
        feats = compute_pit_features(raw)
        maxr = cfg["graph"]["max_reviews_per_product"] if raw_dir.name != "raw_mini" else 60
        g = build_graph(raw, maxr, features=feats)
        g.meta["feature_names"] = FEATURES
        if not args.smoke:
            table = compare_to_published(g, cfg["graph"]["published"], cfg["graph"]["tolerance"])
            print(table.to_string(index=False))
            (ROOT / "outputs" / "tables").mkdir(parents=True, exist_ok=True)
            table.to_csv(ROOT / "outputs" / "tables" / "eda_rebuilt_vs_published.csv", index=False)
            log.note(within_tolerance=bool(table["within_tol"].all()))

    out = processed_path(source, args.smoke)
    save_graph(g, out)
    print(f"{source}: n={g.n} fraud_rate={g.y.mean():.4f} edges={g.edge_count()} -> {out}")
    log.note(n=g.n, fraud_rate=float(g.y.mean()), edges=g.edge_count(), out=str(out))
    log.finish()


if __name__ == "__main__":
    main()
