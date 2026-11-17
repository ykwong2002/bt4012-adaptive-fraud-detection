"""Why the text scorer is at chance around fraud nodes (D22): reads the cached LLM pair judgements
(prompt v2) and the review labels, and splits them by pair type (fraud-fraud, fraud-legit, legit-legit).

Committed tables (no review text):
  pair_case_study_types_raw.csv  per relation and pair type: n, mean fit, share judged fitting, field distance
  pair_case_study_auc_raw.csv    LLM judgement AUC (global and fraud-centred) and single-review field AUC
  pair_case_study_fakes_raw.csv  fraud-legit pairs judged fitting vs not: field profile of each endpoint
Examples with review text go to data/processed/pair_case_study_examples_raw.md (git-ignored).

The judged pairs were sampled across cosine-similarity bins (scripts/04), so shares describe the judged
sample, not all edges. Labels are used for this diagnosis only."""
from __future__ import annotations

import numpy as np
import pandas as pd
from _common import ROOT, parser, processed_path
from sklearn.metrics import roc_auc_score

from camo.data.features_pit import FEATURES
from camo.eval.runlog import RunLog
from camo.llm.score_pairs import current_pairs

TAB = ROOT / "outputs" / "tables"
EVIDENCE = ["specificity", "experiential_evidence", "generic_praise", "promotional_tone", "first_person_detail"]
PROFILE = EVIDENCE + ["named_concretes"]
TEXT_STYLE = ["text_len_log", "caps_ratio", "excl_count"]
FIT = 0.6   # "judged fitting": the top half of the LLM's scale (0.6 and above)


def auc(y, s) -> float:
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan


def load(smoke: bool):
    nodes = processed_path("raw", smoke).with_name(processed_path("raw", smoke).stem + "_nodes.parquet")
    llm = ROOT / "data" / "llm" / ("smoke" if smoke else "")
    if not nodes.exists():
        raise SystemExit("needs the rebuilt graph: scripts/02_build_graph.py --source raw")
    nt = pd.read_parquet(nodes, columns=["review_id", "label", "rating", "text"]).set_index("review_id")
    pit = pd.DataFrame(np.load(processed_path("raw", smoke))["x"], columns=FEATURES, index=nt.index)
    rf = pd.read_parquet(llm / "review_fields.parquet")
    rf = rf[rf["status"] == "ok"].set_index("review_id")
    p = current_pairs(pd.read_parquet(llm / "pair_scores.parquet"))
    p = p[(p["status"] == "ok") & (p["swap"] == 0) & p["u"].isin(rf.index) & p["v"].isin(rf.index)].copy()
    p["label_u"], p["label_v"] = p["u"].map(nt["label"]).to_numpy(), p["v"].map(nt["label"]).to_numpy()
    n_fraud = p["label_u"] + p["label_v"]
    p["type"] = np.select([n_fraud == 2, n_fraud == 1], ["fraud-fraud", "fraud-legit"], "legit-legit")
    p["field_distance"] = np.abs(rf.loc[p["u"], EVIDENCE].to_numpy() - rf.loc[p["v"], EVIDENCE].to_numpy()).mean(1)
    return nt, pit, rf, p


def tables(nt, pit, rf, p):
    types = (p.assign(fitting=p["relative_fit"] >= FIT)
             .groupby(["relation", "type"])
             .agg(n=("relative_fit", "size"), mean_fit=("relative_fit", "mean"), share_fitting=("fitting", "mean"),
                  field_distance=("field_distance", "mean")).reset_index())
    corr = p.groupby("relation").apply(lambda g: np.corrcoef(g["relative_fit"], -g["field_distance"])[0, 1],
                                       include_groups=False)
    types["corr_fit_vs_field_similarity"] = types["relation"].map(corr)

    rows = []
    for r, g in p.groupby("relation"):
        same, fr = (g["label_u"] == g["label_v"]).to_numpy(), (g["type"] != "legit-legit").to_numpy()
        rows.append({"kind": "pair judgement", "relation": r, "name": "relative_fit", "n": len(g),
                     "auc_global": auc(same, g["relative_fit"]), "auc_fraud_centred": auc(same[fr], g["relative_fit"][fr])})
    # how well one review's variable alone separates fraud from legit, either direction
    singles = [("LLM text field", rf[f]) for f in PROFILE]
    singles += [("text style (PIT)" if f in TEXT_STYLE else "behaviour (PIT)", pit[f]) for f in FEATURES]
    for kind, x in singles:
        a = auc(nt["label"].reindex(x.index).to_numpy(), x.to_numpy())
        rows.append({"kind": f"single review: {kind}", "relation": "-", "name": x.name, "n": len(x),
                     "auc_global": max(a, 1 - a)})

    fl = p[p["type"] == "fraud-legit"].copy()
    fraud_is_u = fl["label_u"] == 1
    fl["fraud"], fl["legit"] = np.where(fraud_is_u, fl["u"], fl["v"]), np.where(fraud_is_u, fl["v"], fl["u"])
    fl["judged"] = np.where(fl["relative_fit"] >= FIT, "fitting (edge kept)", "not fitting (edge down-weighted)")
    prof = []
    for j, g in fl.groupby("judged"):
        row = {"judged": j, "n": len(g), "share": len(g) / len(fl)}
        row.update({f"fraud_{f}": rf.loc[g["fraud"], f].mean() for f in PROFILE})
        row.update({f"legit_{f}": rf.loc[g["legit"], f].mean() for f in PROFILE})
        row["fraud_first_review_share"] = pit.loc[g["fraud"], "u_first_review"].mean()
        row["legit_first_review_share"] = pit.loc[g["legit"], "u_first_review"].mean()
        prof.append(row)
    prof.append({"judged": "all fraud reviews (reference)", "n": int(nt["label"].sum()),
                 "fraud_first_review_share": pit.loc[nt["label"] == 1, "u_first_review"].mean()})
    return types, pd.DataFrame(rows), pd.DataFrame(prof), fl


def examples(nt, p, fl, k: int, seed: int = 0) -> str:
    """Random examples (not cherry-picked) from three groups, review text cut to 300 characters."""
    rng = np.random.default_rng(seed)
    groups = [
        ("Fraud-legit pairs judged fitting: camouflage edges the scorer keeps",
         fl[fl["relative_fit"] >= 0.8], "fraud", "legit"),
        ("Fraud-legit pairs judged not fitting: camouflage edges the scorer catches",
         fl[fl["relative_fit"] <= 0.3], "fraud", "legit"),
        ("Fraud-fraud pairs judged not fitting: fraud links the scorer weakens",
         p[(p["type"] == "fraud-fraud") & (p["relative_fit"] <= 0.4)], "u", "v")]
    out = ["# LLM pair judgements: examples (YelpZip slice, prompt v2)", "",
           "Git-ignored: contains review text. Random draws within each group.", ""]
    for title, g, a, b in groups:
        out += [f"## {title} ({len(g)} pairs)", ""]
        for _, r in g.iloc[rng.permutation(len(g))[:k]].iterrows():
            out.append(f"**{r['relation'].upper()}, fit {r['relative_fit']:.1f}.** LLM reason: {r['reason']}")
            for side in (a, b):
                rid = r[side]
                kind = "FRAUD" if nt.loc[rid, "label"] == 1 else "legit"
                out.append(f"- {kind}, {nt.loc[rid, 'rating']:.0f} stars: {str(nt.loc[rid, 'text'])[:300]}")
            out.append("")
    return "\n".join(out)


def main():
    p = parser(__doc__)
    p.add_argument("--examples", type=int, default=6, help="examples per group in the git-ignored file")
    args = p.parse_args()
    tag = f"raw{'_smoke' if args.smoke else ''}"
    log = RunLog(f"05d_pair_case_study_{tag}", vars(args))
    nt, pit, rf, pairs = load(args.smoke)
    types, aucs, fakes, fl = tables(nt, pit, rf, pairs)
    TAB.mkdir(parents=True, exist_ok=True)
    for name, t in (("types", types), ("auc", aucs), ("fakes", fakes)):
        t.to_csv(TAB / f"pair_case_study_{name}_{tag}.csv", index=False)
        print(f"\n{name}:\n{t.round(3).to_string(index=False)}")
    ex = ROOT / "data" / "processed" / f"pair_case_study_examples_{tag}.md"
    ex.write_text(examples(nt, pairs, fl, args.examples))
    print(f"\nexamples (with review text, git-ignored): {ex}")
    log.note(n_pairs=len(pairs)); log.finish()


if __name__ == "__main__":
    main()
