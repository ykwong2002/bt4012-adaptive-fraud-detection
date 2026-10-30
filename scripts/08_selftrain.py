"""Label-scarce response + normal-class bias audit (professor's comment #2).

After the incumbent is trained on the labelled training windows, the later windows form an unlabelled
pool (plus a small labelled probe of `probe_labels_per_window` nodes, the only recent labels a
practitioner has). Pseudo-labels are fused from the incumbent score, Louvain community propensity on
R-U-R ∪ scorer-weighted R-S-R, and AE anomaly rank; a challenger is self-trained for several rounds
under each variant in configs/selftrain.yaml (naive / per-class thresholds / prior matching /
balanced loss / all). Every round is audited for collapse toward the normal class, and the challenger
is promoted only if it beats the incumbent on the probe. Brackets: keep-incumbent (lower) and
retrain-on-all-pool-labels (upper, oracle). Works on the temporal rebuilt graph and, flagged as
`stratified`, on the no-dates .mat graph.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from _common import ROOT, load_config, parser, processed_path, resolve_source

from camo.data.load_mat import load_graph
from camo.eval.runlog import RunLog
from camo.selftrain.loop import run_selftrain

TAB = ROOT / "outputs" / "tables"


def main():
    p = parser(__doc__)
    p.add_argument("--weights", default=None); p.add_argument("--variants", default=None)
    p.add_argument("--seeds", default=None); p.add_argument("--tag", default=None)
    args = p.parse_args()
    cfg, scfg = load_config("base"), load_config("selftrain")
    source = resolve_source(args.source)
    g = load_graph(processed_path(source, args.smoke))
    weights = None
    if args.weights:
        z = np.load(args.weights); weights = {r: z[r] for r in g.edges}
    graph_name = "raw_graph" if weights is None else "refined_soft"
    tag = args.tag or f"{source}{'_smoke' if args.smoke else ''}_{graph_name}"
    log = RunLog(f"08_selftrain_{tag}", {**scfg, **vars(args)})
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else (cfg["smoke"]["seeds"] if args.smoke else cfg["seeds"])
    variants = {k: v for k, v in scfg["variants"].items() if args.variants is None or k in args.variants.split(",")}
    iters = 50 if args.smoke else 300
    rows, brackets = [], []
    for seed in seeds:
        r, b = run_selftrain(g, cfg, scfg, seed, weights, variants, iters, 20 if args.smoke else scfg["probe_labels_per_window"],
                             ae_epochs=50 if args.smoke else 200, graph_name=graph_name)
        rows += r; brackets.append(b)
        print(f"seed {seed} {b['protocol']}: incumbent eval AP={b['lower_keep_incumbent_ap']:.4f} upper={b['upper_all_labels_ap']:.4f} "
              f"prior train={b['train_prior']:.3f} em={b['em_prior']:.3f} true={b['true_pool_prior']:.3f}")
        for v in variants:
            last = [x for x in r if x["variant"] == v][-1]
            print(f"  {v:13s} final: pseudo_pos={last['pseudo_pos_share']:.3f} probe_rec={last['probe_minority_recall']:.3f} "
                  f"chal_eval_ap={last['chal_eval_ap']:.4f} promoted={last['promoted']} collapse={last['collapse']}")
    df = pd.DataFrame(rows); TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"selftrain_rounds_{tag}.csv", index=False)
    pd.DataFrame(brackets).to_csv(TAB / f"selftrain_brackets_{tag}.csv", index=False)
    last = df[df["round"] == df["round"].max()]
    summ = last.groupby("variant").agg(pseudo_pos_share=("pseudo_pos_share", "mean"), prior=("prior", "mean"),
                                       probe_minority_recall=("probe_minority_recall", "mean"), chal_eval_ap=("chal_eval_ap", "mean"),
                                       chal_eval_ap_std=("chal_eval_ap", "std"), inc_eval_ap=("inc_eval_ap", "mean"),
                                       upper_ap=("upper_ap", "mean"), promoted_rate=("promoted", "mean"),
                                       collapse_rate=("collapse", "mean"), recovery_gap=("recovery_gap", "mean")).reset_index()
    summ.to_csv(TAB / f"selftrain_summary_{tag}.csv", index=False)
    print(summ.round(4).to_string(index=False))
    log.note(n_rows=len(df)); log.finish()


if __name__ == "__main__":
    main()
