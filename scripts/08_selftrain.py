"""Label-scarce response + normal-class bias audit (professor's comment #2).

Temporal (rebuilt graph with dates, the reported setting): the post-training windows are walked in time
order. The label-free monitor (scripts/07 logic, calibrated thresholds) raises alarms; at an alarm in
window t a challenger is self-trained from what exists at the end of t only, under each variant in
configs/selftrain.yaml (naive / per-class thresholds / prior matching / balanced loss / all), audited
for collapse toward the normal class every round, and promoted only if it beats the deployed model on
the probe labels collected so far (`probe_labels_per_window` per window). Each window reports the
deployed model, the never-retrained incumbent, the same-settings control retrained without pseudo-labels,
and the all-labels oracle. `--alarm-windows` overrides the monitor (for ablations / smoke tests).

No dates (.mat graph): the static ablation, flagged `stratified`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from _common import ROOT, load_config, parser, processed_path, resolve_source

from camo.data.load_mat import load_graph
from camo.eval.runlog import RunLog
from camo.data.features_pit import FEATURES
from camo.drift.monitor import run_monitor
from camo.eval.splits import equal_count_windows, temporal_plan
from camo.selftrain.loop import run_selftrain, run_triggered

TAB = ROOT / "outputs" / "tables"


def main():
    p = parser(__doc__)
    p.add_argument("--weights", default=None); p.add_argument("--variants", default=None)
    p.add_argument("--seeds", default=None); p.add_argument("--tag", default=None)
    p.add_argument("--alarm-windows", default=None, help="comma list of windows to treat as alarms (overrides the monitor)")
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
    if g.dates is None:
        run_static(g, cfg, scfg, seeds, weights, variants, iters, args, graph_name, tag)
    else:
        run_temporal(g, cfg, scfg, seeds, weights, variants, iters, args, graph_name, tag, source)
    log.finish()


def run_static(g, cfg, scfg, seeds, weights, variants, iters, args, graph_name, tag):
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
                  f"chal_eval_ap={last['chal_eval_ap']:.4f} control_ap={last['control_eval_ap']:.4f} "
                  f"promoted={last['promoted']} collapse={last['collapse']}")
    df = pd.DataFrame(rows); TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"selftrain_rounds_{tag}.csv", index=False)
    pd.DataFrame(brackets).to_csv(TAB / f"selftrain_brackets_{tag}.csv", index=False)
    last = df[df["round"] == df["round"].max()]
    summ = last.groupby("variant").agg(pseudo_pos_share=("pseudo_pos_share", "mean"), prior=("prior", "mean"),
                                       probe_minority_recall=("probe_minority_recall", "mean"), chal_eval_ap=("chal_eval_ap", "mean"),
                                       chal_eval_ap_std=("chal_eval_ap", "std"), control_eval_ap=("control_eval_ap", "mean"),
                                       pseudo_label_gain=("pseudo_label_gain", "mean"), inc_eval_ap=("inc_eval_ap", "mean"),
                                       upper_ap=("upper_ap", "mean"), promoted_rate=("promoted", "mean"),
                                       collapse_rate=("collapse", "mean"), recovery_gap=("recovery_gap", "mean")).reset_index()
    summ.to_csv(TAB / f"selftrain_summary_{tag}.csv", index=False)
    print(summ.round(4).to_string(index=False))


def alarms_from_monitor(g, cfg, w, plan, source, smoke):
    """Alarm per window from the calibrated label-free monitor (same code as scripts/07)."""
    dcfg = load_config("drift")
    proc = ROOT / cfg["paths"]["processed_dir"]
    tag = f"{source}{'_smoke' if smoke else ''}"
    es = proc / f"edge_scores_{tag}.npz"
    scores = None
    if es.exists():
        z = np.load(es); scores = {r: z[r] for r in g.edges}
    names = FEATURES if g.x.shape[1] == len(FEATURES) else [f"f{j}" for j in range(g.x.shape[1])]
    _, _, flags, _, _ = run_monitor(g, w, plan["train"], dcfg, names, scores)
    print(flags.round(4).to_string(index=False))
    return {int(k): bool(a) for k, a in zip(flags["window"], flags["alarm"])}


def run_temporal(g, cfg, scfg, seeds, weights, variants, iters, args, graph_name, tag, source):
    w = equal_count_windows(g.dates, cfg["windows"]["n_windows"])
    plan = temporal_plan(w, cfg["windows"]["n_train_windows"], cfg["windows"]["n_val_windows"])
    if args.alarm_windows:
        forced = {int(k) for k in args.alarm_windows.split(",")}
        alarms = {int(k): k in forced for k in plan["val"] + plan["test"]}
        print("alarms overridden:", sorted(forced))
    else:
        alarms = alarms_from_monitor(g, cfg, w, plan, source, args.smoke)
    win, rounds, summ = [], [], []
    for seed in seeds:
        a, b, c = run_triggered(g, cfg, scfg, seed, weights, variants, iters,
                                20 if args.smoke else scfg["probe_labels_per_window"], alarms, w,
                                ae_epochs=50 if args.smoke else 200, graph_name=graph_name)
        win += a; rounds += b; summ += c
        for s in c:
            print(f"seed {seed} {s['variant']:13s} first_alarm={s['first_alarm']} retrains={s['n_retrains']} "
                  f"promoted={s['n_promoted']} deployed_after={s['deployed_ap_after']:.4f} "
                  f"control_after={s['control_ap_after']:.4f} upper_after={s['upper_ap_after']:.4f}")
    TAB.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(win).to_csv(TAB / f"selftrain_windows_{tag}.csv", index=False)
    pd.DataFrame(rounds).to_csv(TAB / f"selftrain_rounds_{tag}.csv", index=False)
    sm = pd.DataFrame(summ)
    sm.to_csv(TAB / f"selftrain_seeds_{tag}.csv", index=False)
    agg = sm.groupby("variant").mean(numeric_only=True).reset_index()
    agg.to_csv(TAB / f"selftrain_summary_{tag}.csv", index=False)
    print(agg.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
