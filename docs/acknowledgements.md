# Acknowledgements of external code, data and tools

- **Data**: YelpChi (Rayana & Akoglu, KDD 2015); graph version from Dou et al. (CARE-GNN, CIKM 2020).
- **Relation definitions and the >800-reviews filter** follow Liu et al. (GraphConsis, SIGIR 2020)
  and Dou et al. (2020). We re-implemented the relation builder; no code was copied.
- **CARE-static** (`src/camo/models/care_static.py`) is our simplified re-implementation of the
  neighbour-filtering idea in CARE-GNN (fixed similarity threshold, no RL). Published CARE-GNN numbers
  are quoted from the paper, not reproduced from the authors' code.
- **GraphSAGE** (Hamilton et al., 2017): our `WeightedSAGE` layer is written from the paper's mean
  aggregator with edge weights added; PyTorch Geometric is used for tensors and scatter ops only.
- **Behavioural features**: inspired by Rayana & Akoglu Table 2 and Mukherjee et al. (2013); ours are
  recomputed point-in-time and are not a copy of any public feature-extraction script.
- **Libraries**: PyTorch, PyTorch Geometric, scikit-learn, CatBoost, NetworkX (Louvain),
  sentence-transformers (MiniLM embeddings), Anthropic SDK.
- **Generative AI**: Claude (Anthropic) was used (a) as the planned text scorer inside the method
  (`claude-haiku-5-5`, structured outputs, Message Batches); the raw review text never became
  available, so this scorer only ran on the synthetic fixture and **no reported result uses an LLM
  judgement**; the planned LLM-generated camouflage reviews were likewise not produced — the reported
  injection is graph-level and uses no LLM (decisions D13, D18); (b) as a coding assistant (Claude
  Code) for scaffolding, tests, pull-request review and drafts of the documentation. All design
  decisions, code and results were reviewed and are explainable by the group.
