"""Operating-point view for the business-integration section: per test window, what the deployed system
asks of a fraud team at the 0.5 threshold (reviews flagged, fraud caught, reviews checked per fraud
found, analyst labels requested) next to the never-retrained incumbent. One table per dataset, never
mixed: business_raw.csv (YelpZip, natural drift) and business_mat_lvl<L>.csv (.mat loop, clean and
attack scenarios). Means over seeds; reads the per-window tables of scripts/08 and scripts/11."""
from __future__ import annotations

import pandas as pd
from _common import ROOT, parser

from camo.eval.runlog import RunLog

TAB = ROOT / "outputs" / "tables"


def operating_point(w: pd.DataFrame) -> pd.DataFrame:
    keys = ["scenario", "window"] if "scenario" in w else ["window"]
    w = w.assign(probe_labels=w["n_probe"].fillna(0),
        alarm=w["alarm"].astype(float), retrained=w["retrained"].astype(float), promoted=w["promoted"].astype(float),
        flagged=w["deployed_flag_rate"] * w["n_eval"], fraud_caught=w["deployed_recall_at_thr"] * w["n_pos"],
        inc_flagged=w["incumbent_flag_rate"] * w["n_eval"], inc_fraud_caught=w["incumbent_recall_at_thr"] * w["n_pos"])
    cols = {"n_reviews": "n_eval", "n_fraud": "n_pos", "alarm_share": "alarm", "retrained_share": "retrained",
            "promoted_share": "promoted", "flagged": "flagged", "fraud_caught": "fraud_caught",
            "recall": "deployed_recall_at_thr", "precision": "deployed_precision_at_thr", "auc_pr": "deployed_ap",
            "incumbent_flagged": "inc_flagged", "incumbent_fraud_caught": "inc_fraud_caught",
            "incumbent_recall": "incumbent_recall_at_thr", "incumbent_precision": "incumbent_precision_at_thr",
            "labels_used_in_retrain": "probe_labels"}
    if "n_injected" in w:
        cols["n_injected"] = "n_injected"
    t = w.groupby(keys).agg(**{k: (v, "mean") for k, v in cols.items()}).reset_index()
    t["reviews_checked_per_fraud_found"] = 1 / t["precision"]
    return t


def main():
    p = parser(__doc__)
    p.add_argument("--variant", default="per_class", help="self-training variant whose deployed model is reported")
    p.add_argument("--level", default="0.2", help="injection level of the .mat loop table")
    args = p.parse_args()
    log = RunLog("14_business", vars(args))
    inputs = {"raw": TAB / "selftrain_windows_raw_raw_graph.csv",
              f"mat_lvl{args.level}": TAB / f"graph_loop_windows_mat_lvl{args.level}.csv"}
    written = []
    for tag, path in inputs.items():
        if not path.exists():
            print(f"skip {tag}: {path.name} not found")
            continue
        w = pd.read_csv(path)
        t = operating_point(w[w["variant"] == args.variant])
        t.insert(0, "variant", args.variant)
        out = TAB / f"business_{tag}.csv"
        t.to_csv(out, index=False)
        written.append(out.name)
        show = ["window", "alarm_share", "n_reviews", "n_fraud", "flagged", "fraud_caught", "precision",
                "reviews_checked_per_fraud_found", "incumbent_fraud_caught", "labels_used_in_retrain"]
        print(f"\n{tag} ({args.variant}):")
        print(t[(["scenario"] if "scenario" in t else []) + show].round(2).to_string(index=False))
    log.note(written=written); log.finish()


if __name__ == "__main__":
    main()
