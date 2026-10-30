"""Distil the LLM pairwise judgements into an edge scorer, apply it to every edge, write
data/processed/edge_scores_<tag>.npz (per-relation arrays aligned with the graph's edges), the
distillation metrics (with cosine-only / fields-only ablations and the LLM test-retest ceiling) and
the tau-sweep table."""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import ROOT, load_config, parser, processed_path, resolve_source
from camo.data.load_mat import load_graph
from camo.eval.runlog import RunLog
from camo.graph.edge_scorer import fit_and_evaluate, retest_agreement
from camo.graph.embed import embed_texts, mock_embeddings
from camo.graph.rewire import tau_sweep_table
from camo.llm.score_reviews import fields_matrix

TAB = ROOT / "outputs" / "tables"


def main():
    p = parser(__doc__)
    p.add_argument("--mock", action="store_true", help="use mock embeddings (smoke only)")
    args = p.parse_args()
    cfg, rcfg = load_config("base"), load_config("reconstruction")
    source = resolve_source(args.source)
    if source != "raw":
        raise SystemExit("needs the rebuilt graph with text (--source raw)")
    g = load_graph(processed_path(source, args.smoke)); nt = g.node_table
    tag = f"{source}{'_smoke' if args.smoke else ''}"
    llm_dir = ROOT / cfg["paths"]["llm_dir"] / ("smoke" if args.smoke else "")
    log = RunLog(f"05_distil_{tag}", rcfg)
    TAB.mkdir(parents=True, exist_ok=True)

    emb_path = ROOT / cfg["paths"]["processed_dir"] / f"emb_{tag}.npy"
    emb = mock_embeddings(nt["text"].tolist()) if args.mock else embed_texts(nt["text"].tolist(), rcfg["embedding_model"], emb_path)
    rf = pd.read_parquet(llm_dir / "review_fields.parquet")
    fields, fnames = fields_matrix(nt, rf)
    np.save(ROOT / cfg["paths"]["processed_dir"] / f"llm_fields_{tag}.npy", fields)
    print(f"review fields: {int((rf['status']=='ok').sum())}/{len(nt)} scored ok")

    ps = pd.read_parquet(llm_dir / "pair_scores.parquet")
    rid_to_node = pd.Series(np.arange(g.n), index=nt["review_id"].to_numpy())
    ok = ps[(ps["status"] == "ok") & (ps["swap"] == 0)].copy()
    ok["u_node"] = rid_to_node.loc[ok["u"]].to_numpy(); ok["v_node"] = rid_to_node.loc[ok["v"]].to_numpy()
    scorers, metrics = fit_and_evaluate(ok, emb, fields, seed=0, alpha=rcfg["scorer"]["alpha"])
    rt = retest_agreement(ps)
    metrics["llm_retest_agreement_spearman"] = rt["spearman"]; metrics["llm_retest_agreement_n"] = rt["n"]
    metrics.to_csv(TAB / f"distil_metrics_{tag}.csv", index=False)
    print(metrics.round(3).to_string(index=False)); print("test-retest:", rt)

    scores = {}
    for r, e in g.edges.items():
        scores[r] = scorers["all"].predict(e[0], e[1], np.full(e.shape[1], r), emb, fields) if e.shape[1] else np.zeros(0, np.float32)
        print(f"{r}: {e.shape[1]} edges, score mean={scores[r].mean() if e.shape[1] else float('nan'):.3f}")
    out = ROOT / cfg["paths"]["processed_dir"] / f"edge_scores_{tag}.npz"
    np.savez_compressed(out, **scores)
    # ablation scorers too, so 06 can train on cosine-only / fields-only refined graphs
    for sub in ("cos", "fields"):
        np.savez_compressed(str(out).replace(".npz", f"_{sub}.npz"),
                            **{r: scorers[sub].predict(e[0], e[1], np.full(e.shape[1], r), emb, fields) if e.shape[1] else np.zeros(0, np.float32)
                               for r, e in g.edges.items()})
    sweep = tau_sweep_table(g.edges, scores, g.y, rcfg["rewire"]["tau_grid"])
    sweep.to_csv(TAB / f"tau_sweep_{tag}.csv", index=False)
    print(sweep.round(3).to_string(index=False))
    log.note(retest_agreement=rt, out=str(out)); log.finish()


if __name__ == "__main__":
    main()
