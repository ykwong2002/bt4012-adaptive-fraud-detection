# Data sources and verified facts (checked 2026-10-30)

| Fact | Source |
|---|---|
| Graph file: 45,954 nodes, 32 features, 14.53 % fraud (6,677 / 39,277) | https://www.dgl.ai/dgl_docs/generated/dgl.data.FraudYelpDataset.html |
| Relations R-U-R (same user), R-S-R (same product & star), R-T-R (same product & month) | Dou et al., CIKM 2020, https://arxiv.org/abs/2008.08692 ; Liu et al., SIGIR 2020, https://arxiv.org/abs/2005.00625 |
| Undirected edge counts 49,315 / 573,616 / 3,402,743 (DGL reports ×2, both directions) | CARE-GNN Table 2 |
| `.mat` keys only `features, label, net_rur, net_rtr, net_rsr, homo` | https://github.com/YingtongDou/CARE-GNN/blob/master/data_process.py ; DGL `python/dgl/data/fraud.py` |
| Raw YelpChi: 67,395 reviews, 38,063 users, 201 venues, 13.23 % filtered; metadata `(user, product, rating, label, date)` | Rayana & Akoglu, KDD 2015, Table 3; Nash-Detect `Utils/iohelper.py` |
| 45,954 = raw set with products > 800 reviews removed (29,431 users, 182 products) | GraphConsis paper, Sec. 4.1 |
| 32 features = 15 review + 9 user + 8 product features from Rayana Table 2; column order not public | https://github.com/YingtongDou/CARE-GNN/issues/2 |
| Raw text obtainable only by email to the author | https://odds.cs.stonybrook.edu/yelpchi-dataset/ |
| Reference AUC on YelpChi varies 0.54–0.95 with split and tuning | CARE-GNN Table 3; BWGNN (ICML 2022); GADBench (NeurIPS 2023) |
