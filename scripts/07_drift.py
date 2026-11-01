"""Drift detection on the rebuilt graph: label-free monitors (PIT features, LLM fields, raw-graph
text-homophily, refined-vs-raw convergence) per window vs the incumbent's rolling AUC-PR from
scripts/06. Writes outputs/tables/drift_signals_<tag>.csv and drift_summary_<tag>.csv."""
from __future__ import annotations

import numpy as np
import pandas as pd
from _common import ROOT, load_config, parser, processed_path, resolve_source

from camo.data.features_pit import FEATURES
from camo.data.load_mat import load_graph
from camo.drift.monitor import detection_latency, run_monitor
from camo.eval.runlog import RunLog
from camo.eval.splits import equal_count_windows, temporal_plan
from camo.llm.prompts import REVIEW_FIELDS

TAB = ROOT / "outputs" / "tables"


def main():
    p = parser(__doc__)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--model", default="wsage", help="incumbent whose AUC-PR is the labelled comparator")
    p.add_argument("--degrade-rel", type=float, default=0.10, help="relative AUC-PR drop vs val that counts as degradation")
    args = p.parse_args()
    cfg, dcfg = load_config("base"), load_config("drift")
    source = resolve_source(args.source)
    if source != "raw":
        raise SystemExit("drift detection needs dated nodes (--source raw)")
    tag = f"{source}{'_smoke' if args.smoke else ''}"
    g = load_graph(processed_path(source, args.smoke))
    log = RunLog(f"07_drift_{tag}", {**dcfg, **vars(args)})
    proc = ROOT / cfg["paths"]["processed_dir"]
    w = equal_count_windows(g.dates, cfg["windows"]["n_windows"])
    plan = temporal_plan(w, cfg["windows"]["n_train_windows"], cfg["windows"]["n_val_windows"])

    names = FEATURES if g.x.shape[1] == len(FEATURES) else [f"f{j}" for j in range(g.x.shape[1])]
    lf = proc / f"llm_fields_{tag}.npy"
    llm = np.load(lf) if lf.exists() else None
    es = proc / f"edge_scores_{tag}.npz"
    scores = None
    if es.exists():
        z = np.load(es); scores = {r: z[r] for r in g.edges}
    sig, thr, m, h_raw, h_ref = run_monitor(g, w, plan["train"], dcfg, names, scores, llm,
                                            REVIEW_FIELDS + ["llm_unscored"], args.tau)
    if h_raw is not None:
        np.save(proc / f"h_text_{tag}.npy", np.vstack([h_raw, h_ref]))
    print("calibrated thresholds (leave-one-training-window-out):"); print(thr.round(4).to_string(index=False))

    # labelled comparator: incumbent AUC-PR per window from scripts/06 (raw graph)
    res_path = TAB / f"results_{tag}_raw_graph.csv"
    comp = None
    if res_path.exists():
        res = pd.read_csv(res_path)
        r = res[res["model"] == args.model].groupby("fold")["auc_pr"].mean()
        val = r.get("val", np.nan)
        comp = pd.DataFrame({"window": [int(f.split("_w")[1]) for f in r.index if f.startswith("test_w")],
                             "auc_pr": [r[f] for f in r.index if f.startswith("test_w")]})
        comp["auc_pr_val"] = val
        comp["degraded"] = comp["auc_pr"] < val * (1 - args.degrade_rel)
    key = m["monitor"].iloc[0] if len(m) else "none"
    summary = {"monitor": key, **{f"alarm_w{k}": bool(v) for k, v in zip(m["window"], m["alarm"])},
               "n_flagged": int(m["flag"].sum()), "n_windows": len(m)}
    mm = m.merge(comp, on="window", how="left") if comp is not None else m.assign(auc_pr=np.nan, degraded=False)
    if comp is not None:
        test_mask = mm["window"].isin(plan["test"]).to_numpy()
        lat = detection_latency(mm["alarm"].to_numpy()[test_mask], mm["degraded"].fillna(False).astype(bool).to_numpy()[test_mask])
        corr = mm[["psi_mean", "auc_pr"]].corr(method="spearman").iloc[0, 1] if mm["auc_pr"].notna().sum() > 2 else np.nan
        summary.update({f"latency_{k}": v for k, v in lat.items()}); summary["spearman_psi_vs_aucpr"] = corr
    mm.to_csv(TAB / f"drift_monitor_{tag}.csv", index=False)
    thr.to_csv(TAB / f"drift_thresholds_{tag}.csv", index=False)
    print(mm.round(4).to_string(index=False))
    sig.to_csv(TAB / f"drift_signals_{tag}.csv", index=False)
    pd.DataFrame([summary]).to_csv(TAB / f"drift_summary_{tag}.csv", index=False)
    print(sig.round(4).to_string(index=False)); print(summary)
    log.note(**{k: (v if not isinstance(v, (np.bool_, np.integer, np.floating)) else v.item()) for k, v in summary.items()}); log.finish()


if __name__ == "__main__":
    main()
