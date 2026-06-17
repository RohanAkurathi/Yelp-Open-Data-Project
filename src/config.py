"""Central configuration for the recommender and search pipelines.

Everything that controls the size of the experiment lives here so a run can be
reproduced (or scaled up) by editing one file. Defaults are tuned to finish in a
few minutes on a laptop while still using the real Yelp data end to end.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"          # parquet, models, FAISS index (git-ignored)
PLOTS = ROOT / "outputs" / "plots"
REPORTS = ROOT / "outputs" / "reports"

for _d in (ARTIFACTS, PLOTS, REPORTS):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42

# ── Cohort selection ────────────────────────────────────────────────────────────
# The Yelp dump is national, but interactions are concentrated in a handful of
# metros. We restrict to a few well-represented ones and to dining categories so
# the user–item matrix is dense enough for collaborative filtering to mean
# something. Set CITIES = None to let prepare.py pick the busiest cities itself.
CITIES = ["Philadelphia", "Tampa", "Indianapolis", "Nashville", "New Orleans"]
N_CITIES_IF_AUTO = 5

# A business counts as "dining" if its category string mentions any of these.
DINING_CATEGORIES = ("Restaurants", "Food")

# ── k-core filtering ─────────────────────────────────────────────────────────────
# Drop cold users/items until everyone left has at least this many interactions.
# Keeps the matrix dense and the evaluation meaningful.
MIN_USER_REVIEWS = 5
MIN_BUSINESS_REVIEWS = 10

# ── Relevance ────────────────────────────────────────────────────────────────────
# Stars >= this count as a positive ("the user liked this place").
POSITIVE_THRESHOLD = 4.0
# Graded relevance for NDCG: map 1..5 stars onto these gains.
STAR_TO_GAIN = {1.0: 0, 2.0: 0, 3.0: 1, 4.0: 2, 5.0: 3}

# ── Temporal split ───────────────────────────────────────────────────────────────
# Split each cohort by review date so we always predict the future from the past
# (no leakage). Fractions are by chronological order of interactions.
TRAIN_FRAC = 0.70
VALID_FRAC = 0.15      # remainder (0.15) is the held-out test period

# ── Candidate generation (stage 1: ALS) ─────────────────────────────────────────
ALS_FACTORS = 96
ALS_ITERATIONS = 20
ALS_REGULARIZATION = 0.05
ALS_ALPHA = 12.0           # confidence scaling on implicit feedback
N_CANDIDATES = 100         # candidates handed from stage 1 to stage 2 per user

# ── Ranking (stage 2: LightGBM LambdaMART) ───────────────────────────────────────
RANK_EVAL_K = 10           # the K in NDCG@K / Recall@K we report
RANK_TRAIN_USERS = 20_000  # users sampled to train the ranker (those with valid positives)
RANK_EVAL_USERS = 12_000   # users sampled for the held-out test evaluation

# ── Semantic search ──────────────────────────────────────────────────────────────
BASE_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_SAMPLE_REVIEWS = 120_000   # reviews encoded into the FAISS index for the demo
MAX_SEQ_LEN = 128                # truncate reviews to 128 tokens (lighter + faster on MPS)
FINETUNE_PAIRS = 25_000          # (review, same-business review) pairs for fine-tuning
FINETUNE_EPOCHS = 1
FINETUNE_BATCH = 32              # small batches keep training within the M1's unified memory
EMBED_BATCH = 256                # inference encoding has no gradients, so a big batch is fine
