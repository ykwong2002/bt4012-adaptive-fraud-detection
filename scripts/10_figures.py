"""Produce every report figure from the tables in outputs/tables (skips what is missing).
Figures land in outputs/figures/fig_*.png. Pass --tag raw (default) or --tag raw_smoke."""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import ROOT

TAB, FIG = ROOT / "outputs" / "tables", ROOT / "outputs" / "figures"


def read(name):
    p = TAB / name
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:   # e.g. no alarm fired, so no self-training rounds were written
        return None


def headline(tag):
    """Raw vs refined AUC-PR per test window, per model (mean ± std over seeds)."""
    parts = [read(f"results_{tag}_{gname}.csv") for gname in ["raw_graph", "refined_soft", "refined_softrank", "refined_hard_tau0.5"]]
    parts = [p for p in parts if p is not None]
    if not parts:
        return
    df = pd.concat(parts)
    if not df["fold"].str.startswith("test_w").any():      # stratified (no windows): one bar per model and graph
        t = df[df["fold"] == "test"].groupby(["model", "graph"])["auc_pr"].agg(["mean", "std"]).reset_index()
        fig, ax = plt.subplots(figsize=(6, 3.4))
        graphs = sorted(t["graph"].unique()); models = sorted(t["model"].unique()); wdt = 0.8 / len(graphs)
        for i, gname in enumerate(graphs):
            sub = t[t["graph"] == gname].set_index("model").reindex(models)
            ax.bar(np.arange(len(models)) + i * wdt, sub["mean"], wdt, yerr=sub["std"].fillna(0), capsize=3, label=gname)
        ax.set_xticks(np.arange(len(models)) + wdt * (len(graphs) - 1) / 2); ax.set_xticklabels(models)
        ax.set_ylabel("AUC-PR (test)"); ax.set_title(f"{tag}: {df['protocol'].iloc[0]} split, mean ± std over seeds"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(FIG / f"fig_headline_{tag}.png", dpi=150); plt.close(fig)
        return
    df = df[df["fold"].str.startswith("test_w")]
    df["window"] = df["fold"].str.split("_w").str[1].astype(int)
    models = sorted(df["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.4), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), models):
        for gname, sub in df[df["model"] == m].groupby("graph"):
            s = sub.groupby("window")["auc_pr"].agg(["mean", "std"])
            ax.errorbar(s.index, s["mean"], yerr=s["std"].fillna(0), marker="o", capsize=3, label=gname)
        ax.set_title(m); ax.set_xlabel("test window"); ax.set_ylabel("AUC-PR")
        ax.legend(fontsize=8)
    proto = {"waves": "random arrival waves"}.get(df["protocol"].iloc[0], "temporal windows")
    fig.suptitle(f"AUC-PR per test window, {' vs '.join(sorted(df['graph'].unique()))} ({tag}: {proto}, mean ± std over seeds)", y=1.02)
    fig.tight_layout(); fig.savefig(FIG / f"fig_headline_{tag}.png", dpi=150, bbox_inches="tight"); plt.close(fig)


SCORER_COLORS = {"feature": "#1f77b4", "cosine": "#ff7f0e", "rur_profile": "#2ca02c"}
SCORER_LABEL_OFFSET = {"feature": (-8, 8, "right"), "cosine": (8, -12, "left"), "rur_profile": (8, -12, "left")}
MODEL_NAMES = {"wsage": "WeightedSAGE", "catboost": "CatBoost", "care": "CARE-static", "ae": "Autoencoder"}


def scorer_curve(tag):
    """Model AUC-PR against edge-scorer quality: the noisy-oracle sweep with the real scorers placed on it.
    Works for the stratified tag ("mat": fold "test") and the wave protocol ("mat_waves": per seed, the
    mean over test waves). Also writes the plotted numbers to scorer_curve_<tag>.csv."""
    files = [f for f in sorted(TAB.glob(f"results_{tag}_*_softrank.csv"))
             if tag.endswith("waves") or "_waves_" not in f.name]      # keep the two protocols apart
    base = read(f"results_{tag}_raw_graph.csv")
    if not files or base is None:
        return
    source = tag.split("_")[0]

    def per_seed(d, keys):
        d = d[(d["fold"] == "test") | d["fold"].str.startswith("test_w")]
        return d.groupby(keys + ["seed"])["auc_pr"].mean().reset_index()

    df = pd.concat(pd.read_csv(f) for f in files)
    df["scorer"] = df["weights"].str.removeprefix(f"{source}_")
    auc = {}
    for s in df["scorer"].unique():
        # per-seed scorers (fit on one seed's training waves) only belong to the wave protocol
        per_seed_files = sorted(TAB.glob(f"homophily_weighted_{source}_{s}_w*.csv")) if tag.endswith("waves") else []
        hs = [pd.read_csv(f) for f in per_seed_files] or [read(f"homophily_weighted_{source}_{s}.csv")]
        hs = [h for h in hs if h is not None]
        if hs:
            auc[s] = float(np.mean([np.average(h["edge_auc_same_vs_cross"], weights=h["edges"]) for h in hs]))
    df = per_seed(df, ["model", "scorer"])
    df["edge_auc"] = df["scorer"].map(auc)
    t = df.groupby(["model", "scorer", "edge_auc"])["auc_pr"].agg(["mean", "std", "count"]).reset_index()
    if t.empty:          # e.g. the text track: refined_* results but no scorer sweep to place them on
        return
    base = per_seed(base, ["model"])
    raw = base[base["seed"].isin(df["seed"].unique())].groupby("model")["auc_pr"].mean()
    t["raw_graph_auc_pr"] = t["model"].map(raw)
    t.to_csv(TAB / f"scorer_curve_{tag}.csv", index=False)

    models = sorted(t["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(4.4 * len(models), 3.6), sharey=False)
    for ax, m in zip(np.atleast_1d(axes), models):
        sub = t[t["model"] == m]
        noisy = sub[sub["scorer"].str.startswith("noisy")].sort_values("edge_auc")
        ax.errorbar(noisy["edge_auc"], noisy["mean"], yerr=noisy["std"].fillna(0), color="k", lw=2,
                    marker="o", ms=6, capsize=3, label="noisy oracle (simulated)")
        for row in sub[~sub["scorer"].str.startswith("noisy")].itertuples():
            ax.errorbar(row.edge_auc, row.mean, yerr=0 if np.isnan(row.std) else row.std, ls="none", marker="s", ms=8,
                        capsize=3, color=SCORER_COLORS.get(row.scorer, "grey"), label=f"{row.scorer} scorer")
            dx, dy, ha = SCORER_LABEL_OFFSET.get(row.scorer, (8, -12, "left"))
            ax.annotate(row.scorer, (row.edge_auc, row.mean), xytext=(dx, dy), textcoords="offset points", fontsize=8, ha=ha)
        ax.axhline(raw.get(m, np.nan), color="grey", ls="--", lw=1, label="raw graph")
        if m != "catboost" and "catboost" in raw:
            ax.axhline(raw["catboost"], color="grey", ls=":", lw=1, label="CatBoost, raw graph")
        ax.grid(alpha=0.3, lw=0.5)
        ax.set_title(MODEL_NAMES.get(m, m)); ax.set_xlabel("edge-scorer AUC, same- vs cross-label edges"); ax.set_ylabel("AUC-PR (test)")
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(f"How good must an edge scorer be? ({tag}, softrank weights, mean ± std over seeds)", y=1.02)
    fig.tight_layout(); fig.savefig(FIG / f"fig_scorer_curve_{tag}.png", dpi=150, bbox_inches="tight"); plt.close(fig)


SCORER_MARKERS = {"feature": "s", "cosine": "^", "rur_profile": "D"}


def scorer_auc(source, rels=("rsr", "rtr")):
    """Global and fraud-centred edge AUC per scorer on the relations camouflage breaks, edge-weighted;
    per-seed scorers (<name>_w<k>) are averaged over seeds."""
    d = read(f"edge_auc_centred_{source}.csv")
    if d is None:
        return None
    d = d[d["relation"].isin(rels)].copy()
    d["scorer"] = d["scorer"].str.replace(r"_w\d+$", "", regex=True)
    out = {s: {c: np.average(g[c], weights=g["edges"]) for c in ("auc_global", "auc_fraud_centred")}
           for s, g in d.groupby("scorer")}
    return pd.DataFrame.from_dict(out, orient="index")


def scorer_bar(tag="mat_waves", text_source="raw"):
    """Reweighting gain (paired 95 % CI) against edge AUC around fraud nodes (D15). Real scorers are
    drawn at their fraud-centred (filled) and global (hollow) AUC; the YelpZip text scorer is marked on
    the x-axis only, since its model results are on another dataset. Writes scorer_bar_<tag>.csv with
    the curve's predicted gain at each real scorer's position on both axes."""
    from matplotlib.lines import Line2D

    st, auc = read(f"stats_rewiring_{tag}.csv"), scorer_auc(tag.split("_")[0])
    if st is None or auc is None:
        return
    t = st.merge(auc, left_on="scorer", right_index=True)
    text = scorer_auc(text_source)
    text_x = text.loc["text_distilled", "auc_fraud_centred"] if text is not None and "text_distilled" in text.index else None

    rows, models = [], sorted(t["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(4.8 * len(models), 3.9))
    for ax, m in zip(np.atleast_1d(axes), models):
        sub = t[t["model"] == m]
        noisy = sub[sub["scorer"].str.startswith("noisy")].sort_values("auc_fraud_centred")
        cx, cy = noisy["auc_fraud_centred"].to_numpy(), noisy["mean_diff"].to_numpy()
        ax.fill_between(cx, noisy["ci95_low"], noisy["ci95_high"], color="k", alpha=0.12, lw=0)
        ax.plot(cx, cy, color="k", lw=2, marker="o", ms=6)
        for r in sub[~sub["scorer"].str.startswith("noisy")].itertuples():
            c, mk = SCORER_COLORS.get(r.scorer, "grey"), SCORER_MARKERS.get(r.scorer, "s")
            ax.plot([r.auc_global, r.auc_fraud_centred], [r.mean_diff] * 2, color=c, lw=1, alpha=0.6)
            ax.plot(r.auc_global, r.mean_diff, marker=mk, ms=8, mfc="white", mec=c, mew=1.5, ls="none")
            ax.errorbar(r.auc_fraud_centred, r.mean_diff, yerr=[[r.mean_diff - r.ci95_low], [r.ci95_high - r.mean_diff]],
                        marker=mk, ms=8, color=c, capsize=3, ls="none")
            below = r.scorer == "rur_profile"
            ax.annotate(r.scorer, (r.auc_fraud_centred, r.ci95_low if below else r.ci95_high),
                        xytext=(0, -4 if below else 4), textcoords="offset points", fontsize=7, ha="center",
                        va="top" if below else "bottom")
            rows.append({"model": m, "scorer": r.scorer, "auc_global": r.auc_global,
                         "auc_fraud_centred": r.auc_fraud_centred, "gain": r.mean_diff,
                         "ci95_low": r.ci95_low, "ci95_high": r.ci95_high,
                         "curve_gain_at_global": np.interp(r.auc_global, cx, cy),
                         "curve_gain_at_fraud_centred": np.interp(r.auc_fraud_centred, cx, cy)})
        if text_x is not None:
            ax.axvline(text_x, color="k", ls="-.", lw=1)
            ax.annotate("YelpZip text scorer", (text_x, 0.97), xycoords=ax.get_xaxis_transform(), xytext=(-3, 0),
                        textcoords="offset points", rotation=90, va="top", ha="right", fontsize=7)
        ax.axhline(0, color="grey", ls="--", lw=1)
        ax.grid(alpha=0.3, lw=0.5)
        ax.set_title(MODEL_NAMES.get(m, m))
        ax.set_xlabel("edge AUC on R-S-R / R-T-R, edges touching a fraud node")
        ax.set_ylabel("AUC-PR gain over raw graph")
    handles = [Line2D([], [], color="k", lw=2, marker="o", label="noisy oracle (simulated), 95% CI"),
               Line2D([], [], color="grey", marker="s", ls="none", label="real scorer at fraud-centred AUC"),
               Line2D([], [], color="grey", marker="s", mfc="white", ls="none", label="same scorer at global AUC")]
    np.atleast_1d(axes)[0].legend(handles=handles, fontsize=7, loc="upper left", bbox_to_anchor=(0.1, 1.0))
    fig.suptitle(f"Reweighting gain vs edge quality around fraud nodes ({tag}, paired 95% CI over seeds)", y=1.02)
    fig.tight_layout(); fig.savefig(FIG / f"fig_scorer_bar_{tag}.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    out = pd.DataFrame(rows)
    out.to_csv(TAB / f"scorer_bar_{tag}.csv", index=False)
    for axis in ("global", "fraud_centred"):
        err = (out["gain"] - out[f"curve_gain_at_{axis}"]).abs().mean()
        print(f"scorer bar: mean |actual - curve| gain with {axis} AUC = {err:.4f}")


def distil(tag):
    m = read(f"distil_metrics_{tag}.csv")
    if m is None:
        return
    fig, ax = plt.subplots(figsize=(6, 3.2))
    sub = m[m["relation"] == "all"]
    ax.bar(sub["subset"], sub["spearman"], label="distilled scorer vs LLM (held-out pairs)")
    ax.axhline(sub["llm_test_retest_spearman"].iloc[0], color="k", ls="--", label="LLM test-retest ceiling")
    ax.set_ylabel("Spearman"); ax.set_title("Edge-scorer distillation and ablations"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / f"fig_distil_{tag}.png", dpi=150); plt.close(fig)


def tau_sweep(tag):
    t = read(f"tau_sweep_{tag}.csv")
    if t is None:
        return
    t = t[t["tau"] != "soft"].copy(); t["tau"] = t["tau"].astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for r, sub in t.groupby("relation"):
        axes[0].plot(sub["tau"], sub["kept_frac"], marker="o", label=r)
        axes[1].plot(sub["tau"], sub["hetero_frac_dropped"], marker="o", label=r)
    axes[0].set_title("edges kept"); axes[1].set_title("fraction of dropped edges that were cross-label (EDA)")
    for ax in axes:
        ax.set_xlabel("tau"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / f"fig_tau_sweep_{tag}.png", dpi=150); plt.close(fig)


def drift(tag):
    m = read(f"drift_monitor_{tag}.csv")
    if m is None:
        return
    fig, ax = plt.subplots(figsize=(7, 3.4)); ax2 = ax.twinx()
    ax.plot(m["window"], m["psi_mean"], marker="o", color="C0", label=f"PSI, {m['monitor'].iloc[0]} (label-free)")
    if "psi_thr" in m:
        ax.plot(m["window"], m["psi_thr"], color="C0", ls=":", lw=1, label="calibrated threshold")
    ax2.plot(m["window"], m["auc_pr"], marker="s", color="C3", label="incumbent AUC-PR (labelled)")
    for w, a in zip(m["window"], m["alarm"]):
        if a:
            ax.axvline(w, color="k", alpha=0.15, lw=6)
    ax.set_xlabel("window"); ax.set_ylabel("PSI"); ax2.set_ylabel("AUC-PR")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left"); ax.set_title("Drift monitor vs labelled comparator (shaded = alarm)")
    fig.tight_layout(); fig.savefig(FIG / f"fig_drift_{tag}.png", dpi=150); plt.close(fig)


def selftrain(tag):
    for gname in ["raw_graph", "refined_soft"]:
        r = read(f"selftrain_rounds_{tag}_{gname}.csv")
        wins = read(f"selftrain_windows_{tag}_{gname}.csv")
        if r is None and wins is None:
            continue
        temporal = wins is not None
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
        if r is not None and len(r):
            for v, sub in r.groupby("variant"):
                s = sub.groupby("round").agg(pos=("pseudo_pos_share", "mean"), rec=("probe_minority_recall", "mean"))
                axes[0].plot(s.index, s["pos"], marker="o", label=v); axes[1].plot(s.index, s["rec"], marker="o", label=v)
            axes[0].axhline(r["prior"].mean(), color="k", ls="--", lw=1, label="prior")
        axes[0].set_title("pseudo-label positive share"); axes[1].set_title("probe minority recall")
        for ax in axes[:2]:
            ax.set_xlabel("self-training round")
        if temporal:
            # deployed model per window vs the never-retrained incumbent, the no-pseudo-label control and the oracle
            for v, sub in wins.groupby("variant"):
                s = sub.groupby("window")["deployed_ap"].mean(); axes[2].plot(s.index, s.values, marker="o", label=f"deployed ({v})")
            base = wins.groupby("window")[["incumbent_ap", "control_ap", "upper_ap"]].mean()
            axes[2].plot(base.index, base["incumbent_ap"], "k--", lw=1, label="incumbent")
            axes[2].plot(base.index, base["control_ap"], "k-.", lw=1, label="control (no pseudo-labels)")
            axes[2].plot(base.index, base["upper_ap"], "k:", lw=1, label="all labels (oracle)")
            for wdw in wins.loc[wins["alarm"], "window"].unique():
                axes[2].axvline(wdw, color="k", alpha=0.12, lw=6)
            axes[2].set_xlabel("window (shaded = alarm)"); axes[2].set_title("AUC-PR per window")
        else:
            for v, sub in r.groupby("variant"):
                s = sub.groupby("round")["chal_eval_ap"].mean(); axes[2].plot(s.index, s.values, marker="o", label=v)
            axes[2].axhline(r["inc_eval_ap"].mean(), color="k", ls="--", lw=1, label="incumbent")
            if "control_eval_ap" in r:
                axes[2].axhline(r["control_eval_ap"].mean(), color="k", ls="-.", lw=1, label="control (no pseudo-labels)")
            axes[2].axhline(r["upper_ap"].mean(), color="k", ls=":", lw=1, label="all labels (upper)")
            axes[2].set_xlabel("self-training round"); axes[2].set_title("challenger AUC-PR (held-out)")
        for ax in axes:
            ax.legend(fontsize=7)
        fig.suptitle(f"Self-training bias audit ({gname})", y=1.02)
        fig.tight_layout(); fig.savefig(FIG / f"fig_selftrain_{tag}_{gname}.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def adversarial(tag, prefix="adversarial", by="graph"):
    """AUC-PR on all nodes rises with the injected share of fraud (the random-ranker baseline is the
    prevalence), so the plot shows AUC-PR next to that baseline, AUROC, and injected recall@10 % next to
    its chance level (0.10)."""
    s = read(f"{prefix}_summary_{tag}.csv")
    if s is None:
        return
    d = read(f"{prefix}_{tag}.csv")
    prev = d.assign(prev=d["n_pos"] / d["n"]).groupby("level")["prev"].mean() if d is not None and {"n_pos", "n"} <= set(d.columns) else None
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for gname, sub in s.groupby(by):
        axes[0].plot(sub["level"], sub["auc_pr"], marker="o", label=gname)
        axes[1].plot(sub["level"], sub["auroc"], marker="o", label=gname)
        axes[2].plot(sub["level"], sub["injected_recall"], marker="o", label=gname)
    if prev is not None:
        axes[0].plot(prev.index, prev.values, "k--", lw=1, label="prevalence (random ranker)")
    axes[1].axhline(0.5, color="k", ls="--", lw=1, label="chance"); axes[2].axhline(0.10, color="k", ls="--", lw=1, label="chance (top 10 %)")
    axes[0].set_title("AUC-PR, all test nodes"); axes[1].set_title("AUROC, all test nodes"); axes[2].set_title("injected-node recall@10%")
    axes[2].set_ylim(bottom=0)
    for ax in axes:
        ax.set_xlabel("injected fraction of window"); ax.legend(fontsize=7)
    fig.suptitle(f"Camouflage injection into the test windows ({tag}, mean over seeds and windows)", y=1.02)
    fig.tight_layout(); fig.savefig(FIG / f"fig_{prefix}_{tag}.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def graph_loop(tag):
    """End-to-end loop (D16): monitor statistic vs its threshold per wave (clean vs attack), and AUC-PR
    per wave of the deployed model vs incumbent, no-pseudo-label control and all-labels oracle."""
    mon, wins = read(f"graph_loop_monitor_{tag}.csv"), read(f"graph_loop_windows_{tag}.csv")
    if mon is None or wins is None or "deployed_ap" not in wins:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    h = mon[mon["signal"] == "h_text_raw"]
    for scen, sub in h.groupby("scenario"):
        s = sub.groupby("window")[["ks_max", "ks_thr"]].mean()
        axes[0].plot(s.index, s["ks_max"], marker="o", label=f"KS, {scen}")
    axes[0].plot(s.index, s["ks_thr"], "k:", lw=1, label="calibrated threshold")
    axes[0].set_title("monitor: neighbourhood consistency"); axes[0].set_xlabel("wave"); axes[0].legend(fontsize=7)
    att = wins[wins["scenario"] == "attack"]
    for ax, v in zip(axes[1:], ["naive", "per_class"]):        # collapses vs best (bias audit, prof. Q2)
        sub = att[att["variant"] == v]
        if sub.empty:
            continue
        s = sub.groupby("window")[["deployed_ap", "incumbent_ap", "control_ap", "upper_ap"]].mean()
        ax.plot(s.index, s["deployed_ap"], marker="o", lw=2, label=f"deployed ({v})")
        ax.plot(s.index, s["incumbent_ap"], "k--", lw=1, label="incumbent")
        ax.plot(s.index, s["control_ap"], "k-.", lw=1, label="control (no pseudo-labels)")
        ax.plot(s.index, s["upper_ap"], "k:", lw=1, label="all labels (oracle)")
        for wdw in sub.loc[sub["alarm"].astype(bool), "window"].unique():
            ax.axvline(wdw, color="k", alpha=0.08, lw=8)
        ax.set_title(f"attack: AUC-PR per wave, variant {v}"); ax.set_xlabel("wave (shaded = alarm)"); ax.legend(fontsize=7)
    fig.suptitle(f"Graph-only adaptive loop ({tag})", y=1.02)
    fig.tight_layout(); fig.savefig(FIG / f"fig_graph_loop_{tag}.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def graph_loop_rounds(tag):
    """Bias audit of the alarm-triggered self-training (professor's Q2): per variant, the positive share
    of the pseudo-labelled set and the minority recall on the probe labels, round by round, against the
    training prior. Averaged over seeds and retrain windows of the attack scenario."""
    r = read(f"graph_loop_rounds_{tag}.csv")
    if r is None or len(r) == 0:
        return
    r = r[r["scenario"] == "attack"] if "scenario" in r else r
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
    for v, sub in r.groupby("variant"):
        s = sub.groupby("round").agg(pos=("pseudo_pos_share", "mean"), rec=("probe_minority_recall", "mean"))
        axes[0].plot(s.index, s["pos"], marker="o", label=v); axes[1].plot(s.index, s["rec"], marker="o", label=v)
    axes[0].axhline(r["prior"].mean(), color="k", ls="--", lw=1, label=f"training prior ({r['prior'].mean():.3f})")
    axes[0].set_title("positive share of the pseudo-labelled set"); axes[0].set_ylabel("share"); axes[0].set_ylim(0, None)
    axes[1].set_title("minority recall on the probe labels"); axes[1].set_ylabel("recall at the prior's top-k")
    for ax in axes:
        ax.set_xlabel("self-training round"); ax.grid(alpha=0.3, lw=0.5); ax.legend(fontsize=7)
    fig.suptitle(f"Does confidence-based self-training drift toward the normal class? ({tag}, attack scenario)", y=1.02)
    fig.tight_layout(); fig.savefig(FIG / f"fig_graph_loop_rounds_{tag}.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--tag", default="raw"); a = p.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    for fn in (headline, scorer_curve, distil, tau_sweep, drift, selftrain, adversarial):
        fn(a.tag)
    adversarial(a.tag, "graph_adversarial", "model")       # graph-only track (D13), any source
    if a.tag == "raw":
        headline("mat"); headline("mat_waves"); selftrain("mat"); scorer_curve("mat"); scorer_curve("mat_waves"); scorer_bar("mat_waves")
        adversarial("mat", "graph_adversarial", "model")
        # every loop run that exists (level, and noise when it differs from the default 0.1)
        for f in sorted(TAB.glob("graph_loop_windows_mat_lvl*.csv")):
            loop_tag = f.stem.removeprefix("graph_loop_windows_")
            graph_loop(loop_tag); graph_loop_rounds(loop_tag)
    print("figures:", sorted(p.name for p in FIG.glob("fig_*")))


if __name__ == "__main__":
    main()
