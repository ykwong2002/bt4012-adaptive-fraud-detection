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
    return pd.read_csv(p) if p.exists() else None


def headline(tag):
    """Raw vs refined AUC-PR per test window, per model (mean ± std over seeds)."""
    parts = [read(f"results_{tag}_{gname}.csv") for gname in ["raw_graph", "refined_soft", "refined_hard_tau0.5"]]
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
    fig.suptitle("Headline: raw vs text-refined graph, temporal windows", y=1.02)
    fig.tight_layout(); fig.savefig(FIG / f"fig_headline_{tag}.png", dpi=150, bbox_inches="tight"); plt.close(fig)


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
    ax.plot(m["window"], m["psi_mean"], marker="o", color="C0", label="PSI of text-homophily (label-free)")
    ax.axhline(0.2, color="C0", ls=":", lw=1)
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
        if r is None:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
        for v, sub in r.groupby("variant"):
            s = sub.groupby("round").agg(pos=("pseudo_pos_share", "mean"), rec=("probe_minority_recall", "mean"), ap=("chal_eval_ap", "mean"))
            axes[0].plot(s.index, s["pos"], marker="o", label=v); axes[1].plot(s.index, s["rec"], marker="o", label=v)
            axes[2].plot(s.index, s["ap"], marker="o", label=v)
        axes[0].axhline(r["prior"].mean(), color="k", ls="--", lw=1, label="prior")
        axes[2].axhline(r["inc_eval_ap"].mean(), color="k", ls="--", lw=1, label="incumbent")
        axes[2].axhline(r["upper_ap"].mean(), color="k", ls=":", lw=1, label="all labels (upper)")
        axes[0].set_title("pseudo-label positive share"); axes[1].set_title("probe minority recall"); axes[2].set_title("challenger AUC-PR (held-out)")
        for ax in axes:
            ax.set_xlabel("self-training round"); ax.legend(fontsize=7)
        fig.suptitle(f"Self-training bias audit ({gname})", y=1.02)
        fig.tight_layout(); fig.savefig(FIG / f"fig_selftrain_{tag}_{gname}.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def adversarial(tag):
    s = read(f"adversarial_summary_{tag}.csv")
    if s is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for gname, sub in s.groupby("graph"):
        axes[0].plot(sub["level"], sub["auc_pr"], marker="o", label=gname)
        axes[1].plot(sub["level"], sub["injected_recall"], marker="o", label=gname)
    axes[0].set_title("AUC-PR vs injection volume"); axes[1].set_title("injected-node recall@10%")
    for ax in axes:
        ax.set_xlabel("injected fraction of window"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / f"fig_adversarial_{tag}.png", dpi=150); plt.close(fig)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--tag", default="raw"); a = p.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    for fn in (headline, distil, tau_sweep, drift, selftrain, adversarial):
        fn(a.tag)
    if a.tag == "raw":
        headline("mat"); selftrain("mat")
    print("figures:", sorted(p.name for p in FIG.glob("fig_*")))


if __name__ == "__main__":
    main()
