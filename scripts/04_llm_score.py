"""LLM scoring. Stage `reviews`: one structured call per review. Stage `pairs`: relative-fit labels
on sampled training-period edges (+ swapped test-retest subset). Results are cached in data/llm/.

  --dry-run [--limit N]   run N requests synchronously, validate the schema, print the full-set cost
  --submit                submit Message Batches (asks for confirmation unless --yes)
  --collect [--wait]      download ended batches into the cache (--wait polls; run in background)
  --mock                  offline heuristic stand-in, for smoke runs / CI only (never for results)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from _common import ROOT, load_config, parser, processed_path, resolve_source
from camo.data.load_mat import load_graph
from camo.eval.runlog import RunLog
from camo.eval.splits import equal_count_windows, temporal_plan
from camo.graph.embed import embed_texts, mock_embeddings
from camo.llm import score_pairs, score_reviews
from camo.llm.batch_runner import collect, dry_run, estimate_cost, submit
from camo.llm.cache import ParquetCache
from camo.llm.mock import mock_pair_score, mock_review_fields


def client():
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set; copy .env.example to .env (or use --mock for smoke runs)")
    import anthropic

    return anthropic.Anthropic()


def run_stage(args, requests, cache, row_fn, model, state_path, mock_fn=None):
    if not requests:
        print("nothing to do: everything is cached"); return
    print(json.dumps(estimate_cost(requests), indent=1))
    if args.mock:
        cache.add([row_fn(r["custom_id"], mock_fn(r), "ok", "mock") for r in requests])
        print(f"mock-scored {len(requests)} -> {cache.save()}")
    elif args.dry_run:
        c = client()
        res = dry_run(c, requests[: args.limit])
        for cid, data, status, usage in res:
            print(cid, status, usage, json.dumps(data)[:200])
        cache.add([row_fn(cid, data, status, model) for cid, data, status, _ in res])
        print(f"cached {len(res)} dry-run results -> {cache.save()}")
        tok_in = np.mean([u["in"] for *_, u in res]); tok_out = np.mean([u["out"] for *_, u in res])
        print(f"measured mean tokens per request: in={tok_in:.0f} out={tok_out:.0f}")
    elif args.submit:
        if not args.yes and input(f"submit {len(requests)} requests to the Batches API? [y/N] ").lower() != "y":
            raise SystemExit("aborted")
        submit(client(), requests, state_path)
    elif args.collect:
        rows = [row_fn(cid, data, status, model) for cid, data, status in collect(client(), state_path, args.wait)]
        cache.add(rows); print(f"collected {len(rows)} -> {cache.save()}")
        if rows:
            print(pd.Series([r["status"] for r in rows]).value_counts().to_string())
    else:
        print("no action: pass --dry-run, --submit, --collect or --mock")


def main():
    p = parser(__doc__)
    p.add_argument("--stage", choices=["reviews", "pairs", "all"], default="all")
    p.add_argument("--dry-run", action="store_true"); p.add_argument("--limit", type=int, default=20)
    p.add_argument("--submit", action="store_true"); p.add_argument("--yes", action="store_true")
    p.add_argument("--collect", action="store_true"); p.add_argument("--wait", action="store_true")
    p.add_argument("--mock", action="store_true")
    args = p.parse_args()
    cfg, rcfg = load_config("base"), load_config("reconstruction")
    source = resolve_source(args.source)
    if source != "raw":
        raise SystemExit("LLM scoring needs the rebuilt graph with text (--source raw)")
    g = load_graph(processed_path(source, args.smoke))
    nt = g.node_table
    tag = "smoke" if args.smoke else "full"
    llm_dir = ROOT / cfg["paths"]["llm_dir"] / ("smoke" if args.smoke else "")
    log = RunLog(f"04_llm_score_{tag}", {**rcfg, **vars(args)})

    if args.stage in ("reviews", "all"):
        model = rcfg["llm"]["review_model"]
        cache = ParquetCache(llm_dir / "review_fields.parquet", ["review_id"])
        missing = [k[0] for k in cache.missing([(int(r),) for r in nt["review_id"]])]
        reqs = score_reviews.build_review_requests(nt, missing, model)
        txt = nt.set_index("review_id")["text"]
        run_stage(args, reqs, cache, score_reviews.row_from_result, model, llm_dir / "batches" / "reviews.json",
                  mock_fn=lambda r: mock_review_fields(txt.loc[int(r["custom_id"].split("-")[1])]))

    if args.stage in ("pairs", "all"):
        model = rcfg["llm"]["pair_model"]
        emb_path = ROOT / cfg["paths"]["processed_dir"] / f"emb_{source}{'_smoke' if args.smoke else ''}.npy"
        emb = mock_embeddings(nt["text"].tolist()) if args.mock else embed_texts(nt["text"].tolist(), rcfg["embedding_model"], emb_path)
        w = equal_count_windows(g.dates, cfg["windows"]["n_windows"])
        plan = temporal_plan(w, cfg["windows"]["n_train_windows"], cfg["windows"]["n_val_windows"])
        eligible = np.isin(w, plan["train"])
        n_per = 30 if args.smoke else rcfg["llm"]["n_pairs_per_relation"]
        pairs = score_pairs.sample_pairs(g.edges, eligible, emb, n_per, seed=0)
        pairs.to_parquet(llm_dir / "pair_sample.parquet", index=False) if llm_dir.exists() else None
        cache = ParquetCache(llm_dir / "pair_scores.parquet", ["relation", "u", "v", "swap"])
        rid = nt["review_id"].to_numpy()
        keys = [(r.relation, int(rid[r.u_node]), int(rid[r.v_node]), 0) for r in pairs.itertuples()]
        have = cache.have()
        todo = pairs[[k not in have for k in keys]]
        n_retest = 10 if args.smoke else rcfg["llm"]["n_retest_pairs"]
        reqs = score_pairs.build_pair_requests(todo, nt, model, n_retest=n_retest if len(todo) == len(pairs) else 0)
        txt = nt.set_index("review_id")["text"]; rng = np.random.default_rng(0)

        def mock_fn(r):
            _, rel, u, v, swap = r["custom_id"].split("-")
            a, b = (txt.loc[int(u)], txt.loc[int(v)]) if swap == "0" else (txt.loc[int(v)], txt.loc[int(u)])
            return mock_pair_score(a, b, rel, rng)

        run_stage(args, reqs, cache, score_pairs.row_from_result, model, llm_dir / "batches" / "pairs.json", mock_fn)
    log.finish()


if __name__ == "__main__":
    main()
