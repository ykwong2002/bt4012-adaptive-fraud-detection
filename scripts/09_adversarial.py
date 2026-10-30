"""Adversarial validation: inject camouflage reviews (fraud nodes attached to legitimate users) into
the test windows at increasing volumes, keep the training windows untouched, and measure for the raw
vs the text-refined graph: per-window AUC-PR degradation, the label-free monitor's alarm latency, and
(at the highest level, --recover) the self-trained challenger's recovery. Perturbed real legitimate
reviews are the default injection text; --llm-text uses Claude-generated texts from
data/llm/injected_texts.parquet (scored through the same cache as everything else)."""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from _common import ROOT, load_config, parser, processed_path, resolve_source
from camo.adversarial.inject import inject_reviews
from camo.data.features_pit import compute_pit_features
from camo.data.graph_build import build_graph
from camo.data.load_mat import load_graph
from camo.drift.monitor import h_text_raw_and_refined, trigger
from camo.drift.psi import psi
from camo.eval.metrics import evaluate
from camo.eval.protocol import build_folds
from camo.eval.runlog import RunLog
from camo.eval.splits import equal_count_windows, temporal_plan, window_bounds
from camo.graph.embed import embed_texts, mock_embeddings
from camo.llm.mock import mock_review_fields
from camo.llm.score_reviews import fields_matrix, row_from_result
from camo.models.wsage import RelGraph, predict_wsage, train_wsage

TAB = ROOT / "outputs" / "tables"


def main():
    p = parser(__doc__)
    p.add_argument("--levels", default="0,0.05,0.1,0.2"); p.add_argument("--mock", action="store_true")
    p.add_argument("--llm-text", action="store_true"); p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--recover", action="store_true", help="run the self-training challenger at the top level")
    args = p.parse_args()
    cfg, rcfg, dcfg, scfg = load_config("base"), load_config("reconstruction"), load_config("drift"), load_config("selftrain")
    source = resolve_source(args.source)
    if source != "raw":
        raise SystemExit("needs the rebuilt graph (--source raw)")
    tag = f"{source}{'_smoke' if args.smoke else ''}"
    proc = ROOT / cfg["paths"]["processed_dir"]; llm_dir = ROOT / cfg["paths"]["llm_dir"] / ("smoke" if args.smoke else "")
    g0 = load_graph(processed_path(source, args.smoke)); nt0 = g0.node_table
    raw = nt0.drop(columns=[c for c in ["node"] if c in nt0]).copy()
    with open(proc / f"edge_scorer_{tag}.pkl", "rb") as fh:
        scorer = pickle.load(fh)
    rf = pd.read_parquet(llm_dir / "review_fields.parquet")
    llm_texts = pd.read_parquet(llm_dir / "injected_texts.parquet") if args.llm_text else None
    log = RunLog(f"09_adversarial_{tag}", vars(args))
    seeds = cfg["smoke"]["seeds"] if args.smoke else cfg["seeds"][:3]
    maxr = 60 if args.smoke and len(raw) < 1000 else cfg["graph"]["max_reviews_per_product"]

    w0 = equal_count_windows(g0.dates, cfg["windows"]["n_windows"])
    plan = temporal_plan(w0, cfg["windows"]["n_train_windows"], cfg["windows"]["n_val_windows"])
    train_end = g0.dates[np.isin(w0, plan["train"])].max()
    wb = {k: (s, e, n) for k, s, e, n in window_bounds(g0.dates, w0)}
    test_windows = [wb[k] for k in plan["test"]]
    rows, drift_rows = [], []
    for level in [float(v) for v in args.levels.split(",")]:
        for seed in seeds:
            aug, inj = inject_reviews(raw, train_end, test_windows, level, seed, "llm" if args.llm_text else "perturbed", llm_texts)
            feats = compute_pit_features(aug)
            g = build_graph(aug, maxr, features=feats); nt = g.node_table
            emb = mock_embeddings(nt["text"].tolist()) if args.mock else embed_texts(nt["text"].tolist(), rcfg["embedding_model"], None)
            # LLM fields: cached for real reviews; injected perturbed texts inherit their source's fields,
            # LLM-generated texts must be in the cache (scripts/04 on injected_texts) or are mocked
            rf_aug = rf.copy()
            if len(inj):
                extra = []
                src = rf.set_index("review_id")
                for r in inj.itertuples():
                    if not args.llm_text and r.source_review_id in src.index:
                        row = src.loc[r.source_review_id].to_dict(); row["review_id"] = r.review_id; extra.append(row)
                    elif r.review_id in src.index:
                        continue
                    else:
                        extra.append(row_from_result(f"rev-{r.review_id}", mock_review_fields(r.text), "ok", "mock"))
                rf_aug = pd.concat([rf, pd.DataFrame(extra)], ignore_index=True)
            fields, _ = fields_matrix(nt, rf_aug)
            scores = {r: scorer.predict(e[0], e[1], np.full(e.shape[1], r), emb, fields) if e.shape[1] else np.zeros(0, np.float32)
                      for r, e in g.edges.items()}
            w = equal_count_windows(g0.dates, cfg["windows"]["n_windows"])  # window ids from the un-injected calendar
            # injected nodes are assigned to the window whose date range contains them
            node_w = np.full(g.n, -1)
            for k, (s, e, _) in wb.items():
                node_w[(g.dates >= s) & (g.dates <= e)] = k
            inj_mask = (nt["injected"].to_numpy() == 1) if "injected" in nt else np.zeros(g.n, bool)
            for graph_name, weights in [("raw_graph", None), ("refined_soft", scores)]:
                _, train_idx, folds = build_folds(g, cfg, seed, weights, "soft" if weights else None)
                x = ((g.x - g.x[train_idx].mean(0)) / (g.x[train_idx].std(0) + 1e-6)).astype(np.float32)
                gtr = RelGraph(g.n, folds[0].edges, folds[0].weights); gva = RelGraph(g.n, folds[1].edges, folds[1].weights)
                model = train_wsage(x, g.y, gtr, train_idx, folds[1].eval_idx, gva, seed, epochs=30 if args.smoke else 150)
                for f in folds[2:]:
                    s = predict_wsage(model, x, RelGraph(g.n, f.edges, f.weights), seed)
                    m = evaluate(g.y[f.eval_idx], s[f.eval_idx])
                    k = int(f.name.split("_w")[1]); ninj = int(inj_mask[f.eval_idx].sum())
                    # recall on injected nodes specifically (how many camouflaged injections are caught in top 10%)
                    top = f.eval_idx[np.argsort(-s[f.eval_idx])[: max(1, len(f.eval_idx) // 10)]]
                    inj_rec = float(inj_mask[top].sum() / max(1, ninj)) if ninj else np.nan
                    rows.append({"level": level, "seed": seed, "graph": graph_name, "window": k, "n_injected": ninj,
                                 "injected_recall@10%": inj_rec, **m})
            # label-free monitor on the raw graph with dated neighbours
            h_raw, _ = h_text_raw_and_refined(g.edges, scores, g.n, g.dates, args.tau)
            ref = np.isin(node_w, plan["train"])
            flags = []
            for k in plan["test"]:
                cur = node_w == k
                ps = psi(h_raw[ref], h_raw[cur], dcfg["psi_bins"])
                flags.append(ps > dcfg["psi_threshold"])
                drift_rows.append({"level": level, "seed": seed, "window": k, "psi_h_text": ps,
                                   "mean_h_text": float(np.nanmean(h_raw[cur])), "n_injected": int(inj_mask[cur].sum())})
            al = trigger(np.array(flags), dcfg["k_consecutive"])
            for i, k in enumerate(plan["test"]):
                drift_rows[-len(plan["test"]) + i]["alarm"] = bool(al[i])
            print(f"level {level} seed {seed}: injected={len(inj)} alarm windows={[k for k, a in zip(plan['test'], al) if a]}")
        if args.recover and level == max(float(v) for v in args.levels.split(",")):
            from camo.selftrain.loop import run_selftrain
            rec_rows = []
            for seed in seeds:
                r, b = run_selftrain(g, cfg, scfg, seed, scores, {"all": scfg["variants"]["all"]}, 50 if args.smoke else 300,
                                     20 if args.smoke else scfg["probe_labels_per_window"], ae_epochs=50 if args.smoke else 200, graph_name="refined_soft")
                rec_rows += r
            pd.DataFrame(rec_rows).to_csv(TAB / f"adversarial_recovery_{tag}.csv", index=False)
    df = pd.DataFrame(rows); TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"adversarial_{tag}.csv", index=False)
    pd.DataFrame(drift_rows).to_csv(TAB / f"adversarial_drift_{tag}.csv", index=False)
    summ = df.groupby(["level", "graph"]).agg(auc_pr=("auc_pr", "mean"), auroc=("auroc", "mean"),
                                              injected_recall=("injected_recall@10%", "mean")).reset_index()
    summ.to_csv(TAB / f"adversarial_summary_{tag}.csv", index=False)
    print(summ.round(4).to_string(index=False))
    log.finish()


if __name__ == "__main__":
    main()
