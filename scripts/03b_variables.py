"""Variable summary for the report (text track): every record field, relation, point-in-time feature and
LLM evidence field with its type, meaning, where it is used, how it is kept leak-free, and the
legit-vs-fraud effect size. Writes outputs/tables/variables_<tag>.csv.

PIT effect sizes come from eda_features_<tag>.csv (scripts/03). LLM-field effect sizes need the rebuilt
graph's node table (scripts/02 --source raw); without it those columns are left empty."""
from __future__ import annotations

import numpy as np
import pandas as pd
from _common import ROOT, load_config, parser, processed_path

from camo.data.features_pit import FEATURES
from camo.eval.runlog import RunLog
from camo.llm.prompts import REVIEW_FIELDS

TAB = ROOT / "outputs" / "tables"

PIT_LEAK = "uses only reviews dated on or before this one (tests/test_features_pit.py)"
LLM_LEAK = "from this review's own text only; never asked for a fraud verdict"

# group, variable, type, description, used_by, leakage control
VARIABLES = [
    ("record", "review_id", "id", "row of the metadata file; join key between metadata and review text",
     "join", "strict join, raises on any unmatched review"),
    ("record", "user_id", "categorical", "reviewer account", "R-U-R relation, user features", "-"),
    ("record", "product_id", "categorical", "restaurant", "R-S-R / R-T-R relations, product features", "-"),
    ("record", "rating", "ordinal 1-5", "star rating", "R-S-R relation, rating features", "-"),
    ("record", "date", "date (day)", "posting date", "temporal windows, point-in-time cutoffs", "-"),
    ("record", "label", "binary", "1 = filtered by Yelp's recommender (release label -1), 0 = recommended",
     "target only", "never seen by edge scorers or the drift monitor"),
    ("record", "text", "free text", "review body", "LLM fields, embeddings, 3 style features", "-"),
    ("relation", "R-U-R", "edge", "two reviews by the same user", "all graph models", "edge visible once both reviews exist"),
    ("relation", "R-S-R", "edge", "same restaurant, same star rating", "all graph models", "edge visible once both reviews exist"),
    ("relation", "R-T-R", "edge", "same restaurant, same month", "all graph models", "edge visible once both reviews exist"),
    ("pit", "u_n_prev", "count", "user's earlier reviews", "node features", PIT_LEAK),
    ("pit", "u_mean_rating_prev", "continuous", "user's mean rating over earlier reviews", "node features", PIT_LEAK),
    ("pit", "u_rating_dev_prev", "continuous", "|rating - user's earlier mean|", "node features", PIT_LEAK),
    ("pit", "u_days_since_prev", "continuous", "days since the user's previous review (3,650 if none)", "node features", PIT_LEAK),
    ("pit", "u_sameday_n", "count", "user's reviews posted on the same day", "node features", PIT_LEAK),
    ("pit", "u_first_review", "binary", "1 if this is the user's first review", "node features", PIT_LEAK),
    ("pit", "p_n_prev", "count", "restaurant's earlier reviews", "node features", PIT_LEAK),
    ("pit", "p_mean_rating_prev", "continuous", "restaurant's mean rating over earlier reviews", "node features", PIT_LEAK),
    ("pit", "rating_dev_p", "continuous", "rating - restaurant's earlier mean", "node features", PIT_LEAK),
    ("pit", "rating_extreme", "binary", "1 if the rating is 1 or 5 stars", "node features", PIT_LEAK),
    ("pit", "p_rate_30d", "count", "restaurant's reviews in the previous 30 days", "node features", PIT_LEAK),
    ("pit", "text_len_log", "continuous", "log(1 + word count)", "node features", PIT_LEAK),
    ("pit", "caps_ratio", "continuous", "share of letters that are capitals", "node features", PIT_LEAK),
    ("pit", "excl_count", "count", "number of exclamation marks", "node features", PIT_LEAK),
    ("llm", "specificity", "score 0-1", "concrete, checkable details (dishes, prices, staff names)", "edge scorer", LLM_LEAK),
    ("llm", "experiential_evidence", "score 0-1", "a first-hand visit is narrated", "edge scorer", LLM_LEAK),
    ("llm", "generic_praise", "score 0-1", "interchangeable superlatives", "edge scorer", LLM_LEAK),
    ("llm", "promotional_tone", "score 0-1", "reads like marketing copy", "edge scorer", LLM_LEAK),
    ("llm", "first_person_detail", "score 0-1", "personal context (company, reason for visit)", "edge scorer", LLM_LEAK),
    ("llm", "named_concretes", "count", "distinct named concrete items", "edge scorer", LLM_LEAK),
    ("llm", "exclamation_heavy", "binary", "style flag", "edge scorer", LLM_LEAK),
    ("llm", "all_caps_words", "binary", "style flag", "edge scorer", LLM_LEAK),
    ("llm", "template_like", "binary", "reads like a filled template", "edge scorer", LLM_LEAK),
]


def effect_sizes(x: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Legit and fraud means and Cohen's d, the same definition as scripts/03."""
    rows = []
    for c in x.columns:
        a, b = x.loc[y == 0, c].astype(float), x.loc[y == 1, c].astype(float)
        rows.append({"variable": c, "mean_legit": a.mean(), "mean_fraud": b.mean(),
                     "cohen_d": (b.mean() - a.mean()) / (np.sqrt((a.var() + b.var()) / 2) + 1e-9)})
    return pd.DataFrame(rows)


def llm_effect_sizes(smoke: bool) -> pd.DataFrame | None:
    nodes = processed_path("raw", smoke).with_name(processed_path("raw", smoke).stem + "_nodes.parquet")
    cache = ROOT / "data" / "llm" / ("smoke" if smoke else "") / "review_fields.parquet"
    if not nodes.exists() or not cache.exists():
        return None
    nt = pd.read_parquet(nodes, columns=["review_id", "label"])
    f = pd.read_parquet(cache)
    m = nt.merge(f[f["status"] == "ok"][["review_id"] + REVIEW_FIELDS], on="review_id", how="inner")
    return effect_sizes(m[REVIEW_FIELDS], m["label"].to_numpy())


def main():
    args = parser(__doc__).parse_args()
    tag = f"raw{'_smoke' if args.smoke else ''}"
    log = RunLog(f"03b_variables_{tag}", vars(args))
    assert [v for g, v, *_ in VARIABLES if g == "pit"] == FEATURES, "PIT list out of sync with features_pit.FEATURES"
    assert [v for g, v, *_ in VARIABLES if g == "llm"] == REVIEW_FIELDS, "LLM list out of sync with prompts.REVIEW_FIELDS"

    table = pd.DataFrame(VARIABLES, columns=["group", "variable", "type", "description", "used_by", "leakage_control"])
    dcfg = load_config("drift")
    if "features_pit" in dcfg["signals"]:
        pit = table["group"] == "pit"
        watched = pit & ~table["variable"].isin(dcfg["exclude_features"])
        table.loc[watched, "used_by"] += ", drift monitor"
        table.loc[pit & ~watched, "used_by"] += " (not monitored: grows with time, D8)"
    if "llm_fields" in dcfg["signals"]:
        table.loc[table["group"] == "llm", "used_by"] += ", drift monitor"
    stats = []
    pit = TAB / f"eda_features_{tag}.csv"
    if pit.exists():
        stats.append(pd.read_csv(pit).rename(columns={"feature": "variable"})[["variable", "mean_legit", "mean_fraud", "cohen_d"]])
    llm = llm_effect_sizes(args.smoke)
    if llm is not None:
        stats.append(llm)
    else:
        print("LLM-field effect sizes skipped: build the raw graph first (scripts/02_build_graph.py --source raw)")
    if stats:
        table = table.merge(pd.concat(stats), on="variable", how="left")

    TAB.mkdir(parents=True, exist_ok=True)
    out = TAB / f"variables_{tag}.csv"
    table.to_csv(out, index=False)
    print(table.drop(columns=["description", "leakage_control"]).round(3).to_string(index=False))
    log.note(out=str(out), llm_effect_sizes=llm is not None); log.finish()


if __name__ == "__main__":
    main()
