"""End-to-end adaptive loop on the graph-only track (decisions D16): monitor -> alarm -> label-scarce
retraining -> promotion, on any graph, including the public .mat.

Per seed, nodes get random arrival waves (D13; train 0-2, val 3, test 4-7). Two scenarios:
  clean   no injection: every alarm is a false alarm (the monitor's null behaviour);
  attack  camouflaged fraud nodes (D13 injection) appended to every wave from --attack-from on.
The label-free monitor (D8, calibrated on leave-one-training-wave-out comparisons) watches two
signals: the node features, and h_text_raw, each node's mean edge-consistency with neighbours that
arrived in the same wave, where edge consistency is the cosine of training-standardised features
(the label-free scorer with the best fraud-centred quality, D15). Same-wave neighbours only, because
with all earlier neighbours a later node averages over more of them, its h is less noisy and the
distribution narrows wave by wave: the monitor would alarm on graph growth (cf. D14). Its alarms drive the alarm-triggered,
time-respecting self-training of D9 (all five bias-audit variants), compared per wave against the
never-retrained incumbent, a no-pseudo-label control and an all-labels oracle.

  uv run python scripts/11_graph_loop.py --source mat [--level 0.2] [--attack-from 4] [--seeds 0,1,2]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from _common import ROOT, load_config, parser, processed_path, resolve_source

from camo.adversarial.graph_inject import inject_camouflage
from camo.data.load_mat import Graph, load_graph
from camo.drift.monitor import run_monitor
from camo.eval.runlog import RunLog
from camo.eval.splits import temporal_plan
from camo.eval.waves import assign_waves
from camo.selftrain.loop import run_triggered

TAB = ROOT / "outputs" / "tables"


def cosine_consistency(x: np.ndarray, edges: dict, train_idx: np.ndarray) -> dict:
    """Label-free edge consistency in [0, 1]: cosine of features standardised with training-node
    statistics only, so a score never depends on a node that arrives later."""
    xs = (x - x[train_idx].mean(0)) / (x[train_idx].std(0) + 1e-6)
    xs /= np.linalg.norm(xs, axis=1, keepdims=True) + 1e-9
    return {r: ((np.einsum("ij,ij->i", xs[e[0]], xs[e[1]]) + 1) / 2).astype(np.float32) for r, e in edges.items()}


def per_signal_flags(sig: pd.DataFrame, thr: pd.DataFrame) -> pd.DataFrame:
    cur = sig[~sig["null"]].merge(thr, on="signal", how="left")
    cur["flag"] = (cur["psi_mean"] > cur["psi_thr"]) | (cur["ks_max"] > cur["ks_thr"])
    return cur[["window", "signal", "psi_mean", "psi_thr", "ks_max", "ks_thr", "mean_ref", "mean_cur", "flag"]]


def main():
    p = parser(__doc__)
    p.add_argument("--level", type=float, default=0.2, help="injected nodes per wave, as a fraction of the wave")
    p.add_argument("--attack-from", type=int, default=4, help="first wave that receives injections")
    p.add_argument("--seeds", default=None, help="comma list; default first 3 config seeds (1 for --smoke)")
    p.add_argument("--variants", default=None, help="comma list of self-training variants (default: all)")
    p.add_argument("--scenarios", default="clean,attack")
    p.add_argument("--monitor-only", action="store_true", help="skip self-training (fast monitor check)")
    args = p.parse_args()
    cfg, dcfg, scfg = load_config("base"), load_config("drift"), load_config("selftrain")
    source = resolve_source(args.source)
    tag = f"{source}{'_smoke' if args.smoke else ''}_lvl{args.level}"
    g0 = load_graph(processed_path(source, args.smoke))
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else cfg["smoke"]["seeds"] if args.smoke else cfg["seeds"][:3])
    variants = {k: v for k, v in scfg["variants"].items() if args.variants is None or k in args.variants.split(",")}
    iters = 50 if args.smoke else 300
    probe = 20 if args.smoke else scfg["probe_labels_per_window"]
    names = [f"f{j}" for j in range(g0.x.shape[1])]
    log = RunLog(f"11_graph_loop_{tag}", {**vars(args), "drift": dcfg, "selftrain": scfg})

    mon_rows, win_rows, round_rows, summ_rows = [], [], [], []
    for seed in seeds:
        w0 = assign_waves(g0.y, cfg["windows"]["n_windows"], seed)
        plan = temporal_plan(w0, cfg["windows"]["n_train_windows"], cfg["windows"]["n_val_windows"])
        attacked = [k for k in plan["test"] if k >= args.attack_from]
        for scen in args.scenarios.split(","):
            level = args.level if scen == "attack" else 0.0
            g, w, inj = inject_camouflage(g0, w0, plan["train"], attacked, level, seed)
            train_idx = np.where(np.isin(w, plan["train"]))[0]
            scores = cosine_consistency(g.x, g.edges, train_idx)
            same = {r: w[e[0]] == w[e[1]] for r, e in g.edges.items()}
            g_mon = Graph(g.x, g.y, {r: e[:, same[r]] for r, e in g.edges.items()}, g.dates)
            sig, thr, flags, _, _ = run_monitor(g_mon, w, plan["train"], dcfg, names,
                                                {r: s_[same[r]] for r, s_ in scores.items()})
            n_inj = {int(k): int(inj[w == k].sum()) for k in np.unique(w)}
            for r in per_signal_flags(sig, thr).itertuples(index=False):
                mon_rows.append({"scenario": scen, "seed": seed, **r._asdict(), "n_injected": n_inj[int(r.window)]})
            alarms = {int(k): bool(a) for k, a in zip(flags["window"], flags["alarm"])}
            print(f"seed {seed} {scen:6s}: injected={int(inj.sum())} flagged={flags.loc[flags['flag'], 'window'].tolist()} "
                  f"alarms={[k for k, a in alarms.items() if a]}")
            if args.monitor_only:
                win_rows += [{"scenario": scen, "seed": seed, "window": k, "alarm": a, "variant": "-"} for k, a in alarms.items()]
                continue
            wr, rr, sm = run_triggered(g, cfg, scfg, seed, None, variants, iters, probe, alarms, w,
                                       ae_epochs=50 if args.smoke else 200, graph_name="raw_graph")
            for rows, out in ((wr, win_rows), (rr, round_rows), (sm, summ_rows)):
                for r in rows:
                    r.update(scenario=scen, level=level)
                    if "window" in r and r["window"] is not None and "n_eval" in r:
                        r["n_injected"] = n_inj[int(r["window"])]
                    out.append(r)

    TAB.mkdir(parents=True, exist_ok=True)
    mon = pd.DataFrame(mon_rows); mon.to_csv(TAB / f"graph_loop_monitor_{tag}.csv", index=False)
    wins = pd.DataFrame(win_rows); wins.to_csv(TAB / f"graph_loop_windows_{tag}.csv", index=False)
    pd.DataFrame(round_rows).to_csv(TAB / f"graph_loop_rounds_{tag}.csv", index=False)
    sm = pd.DataFrame(summ_rows)
    if not args.monitor_only:
        sm.to_csv(TAB / f"graph_loop_seeds_{tag}.csv", index=False)

    # monitor: per scenario and signal, how often a monitored post-training wave is flagged / alarmed
    post = plan["val"] + plan["test"]
    alarm_by = wins.drop_duplicates(["scenario", "seed", "window"]).set_index(["scenario", "seed", "window"])["alarm"]
    mon["alarm"] = [bool(alarm_by.get((s, sd, k), False)) for s, sd, k in zip(mon["scenario"], mon["seed"], mon["window"])]
    mon["attacked"] = mon["n_injected"] > 0
    msum = (mon[mon["window"].isin(post)].groupby(["scenario", "signal", "attacked"])
               .agg(waves=("flag", "size"), flag_rate=("flag", "mean"), alarm_rate=("alarm", "mean"),
                    psi=("psi_mean", "mean"), ks=("ks_max", "mean"), mean_cur=("mean_cur", "mean")).reset_index())
    msum.to_csv(TAB / f"graph_loop_monitor_summary_{tag}.csv", index=False)
    print(msum.round(4).to_string(index=False))
    if args.monitor_only:
        log.finish(); return
    summ = sm.groupby(["scenario", "variant"]).mean(numeric_only=True).reset_index()
    summ.to_csv(TAB / f"graph_loop_summary_{tag}.csv", index=False)
    print(summ.round(4).to_string(index=False))
    log.note(n_windows=len(wins)); log.finish()


if __name__ == "__main__":
    main()
