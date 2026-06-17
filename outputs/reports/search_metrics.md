# Semantic search — encoder fine-tuning

Same-business retrieval on **1,500 held-out queries** from **2,053 businesses not seen during fine-tuning**.

| encoder | Hit@10 | MRR@10 |
| --- | --- | --- |
| MiniLM (base) | 0.1693 | 0.1027 |
| fine-tuned | 0.2247 | 0.1376 |

Fine-tuning lifts Hit@10 by **32.7%** and MRR@10 by **34.0%** on unseen businesses.
