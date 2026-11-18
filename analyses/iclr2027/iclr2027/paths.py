"""Paths, pinned data hashes and import setup shared by every script.

All locations come from environment variables, with repo-relative defaults:

  ROAR_DATA         directory holding the Hugging Face mirror's *.parquet tables
                    (default: <repo>/data/roar_hf/data)
  HARNESS_GEN_DATA  root of the harness-generalization runs of Gupta et al. (2026)
                    (contains metadata/runs.parquet and data/...); RQ4 external corpus
  ROAR_RAW_DIR      raw run directories of the submitted SkyDiscover campaigns
                    (only for the two scripts that re-read raw files)
  RUNS_DB    SQLite run database used to match campaigns to raw directories
                    (only for the parser validation)
  ICLR_RESULTS_DIR, ICLR_FIGURES_DIR  output directories (default: results/, figures/ here)
  ROAR_BACKEND      "parquet" (default): ROAR's DB query functions are replaced by
                    pandas mirrors over ROAR_DATA; "postgres": ROAR's own SQL against
                    DATABASE_URL.
"""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = os.path.dirname(HERE)                       # analyses/iclr2027
REPO_ROOT = os.path.dirname(os.path.dirname(ANALYSIS_DIR))  # repository root
ROAR_SRC = os.path.join(REPO_ROOT, "src")
RESULTS = os.environ.get("ICLR_RESULTS_DIR", os.path.join(ANALYSIS_DIR, "results"))
FIGURES = os.environ.get("ICLR_FIGURES_DIR", os.path.join(ANALYSIS_DIR, "figures"))
DERIVED = os.path.join(ANALYSIS_DIR, "data")

ROAR_DATA = os.environ.get("ROAR_DATA", os.path.join(REPO_ROOT, "data", "roar_hf", "data"))
HARNESS_GEN_DATA = os.environ.get("HARNESS_GEN_DATA", os.path.join(REPO_ROOT, "data", "harness_gen"))
ROAR_RAW_DIR = os.environ.get("ROAR_RAW_DIR", "")
RUNS_DB = os.environ.get("RUNS_DB", "")
QUICK = os.environ.get("QUICK", "0") == "1"

# SHA-256 of the Hugging Face mirror tables the paper's numbers were computed on.
PARQUET_SHA256 = {
    "artifacts.parquet": "206889ad70ee56d16d0aa1e4ccecd52fed8b3d7803e729636ef4ec979f3dff57",
    "campaigns.parquet": "6dca8718d68c14306c5e20ec418778bd6f14595f9f9205a736baa4e742638031",
    "candidate_edges.parquet": "741144eeb126cf780149f2cec596cd22e47534142cf3c45d27066f618a8e890e",
    "candidates.parquet": "1f2a0cd56f5b763582cfc4a07efd9b681fd827634cc971e4c0f7ee5e93370ae0",
    "measurements.parquet": "ca0c5d810190a8c475ba1ffcc5b44663b50e6298df5289e7fed581a9503d11d6",
    "systems.parquet": "4df3006c02899aab2525bb78939f9a16e0a7ee7f2276580eb149ab5d62d5fe0e",
}
# Submitted corpus = campaigns with this author (928 campaigns).
SUBMITTED_AUTHOR = "Roar Authors"
POST_SUBMISSION_AUTHOR = "Roar Authors 2"

os.environ.setdefault("MPLBACKEND", "Agg")
if ROAR_SRC not in sys.path:
    sys.path.insert(0, ROAR_SRC)
if ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, ANALYSIS_DIR)


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


_VERIFIED = False


def verify_parquet():
    """Abort unless every mirror table matches its pinned SHA-256."""
    global _VERIFIED
    if _VERIFIED:
        return
    for name, want in PARQUET_SHA256.items():
        p = os.path.join(ROAR_DATA, name)
        if not os.path.exists(p):
            sys.exit(f"missing {p}; set ROAR_DATA to the directory holding the mirror's parquet files")
        got = sha256(p)
        if got != want:
            sys.exit(f"SHA-256 mismatch for {name}: {got} != {want}")
    _VERIFIED = True


def out(*parts):
    p = os.path.join(*parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p
