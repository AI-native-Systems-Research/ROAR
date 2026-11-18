"""Per-candidate code length for the RQ3 length control.

The code artifact of a candidate is chosen as src/export_for_embedding.py does when it
selects what to embed: an artifact whose uri contains '/programs/' or has a source-file
extension; '/programs/' files first, then the largest size_bytes.
  art_bytes = artifacts.size_bytes of that artifact (from the mirror; available for all
              candidates, but for program JSON files it also counts the stored prompt);
  tokens    = tiktoken o200k_base token count of the code text, as loaded by ROAR's
              export_for_embedding.load_code_from_uri from the raw file. The raw files are
              not part of the parquet mirror, so the counts are stored in
              data/code_tokens.csv.gz, produced by scripts/rq3_extract_code_tokens.py.
"""
import os

import pandas as pd

from . import paths, parquet_backend as pb

EXT = (".rs", ".py", ".cpp", ".c", ".java", ".js", ".ts", ".go")
TOKENS_FILE = os.path.join(paths.DERIVED, "code_tokens.csv.gz")


def code_artifacts():
    paths.verify_parquet()
    a = pd.read_parquet(os.path.join(paths.ROAR_DATA, "artifacts.parquet"),
                        columns=["candidate_id", "uri", "size_bytes", "content_hash"])
    ids = set(pb.tables()["candidates"].id)
    a = a[a.candidate_id.isin(ids)]
    prog = a.uri.str.contains("/programs/", regex=False)
    a = a[prog | a.uri.str.endswith(EXT)].copy()
    a["prio"] = (~a.uri.str.contains("/programs/", regex=False)).astype(int)
    a["sz"] = a.size_bytes.fillna(0)
    a = a.sort_values(["candidate_id", "prio", "sz"], ascending=[True, True, False], kind="mergesort")
    a = a.drop_duplicates("candidate_id", keep="first")
    return a.rename(columns={"size_bytes": "art_bytes"})[["candidate_id", "uri", "art_bytes", "content_hash"]]


def lengths():
    """candidate_id -> art_bytes, tokens (NaN where the code text was not available)."""
    a = code_artifacts().set_index("candidate_id")[["art_bytes"]].astype(float)
    t = pd.read_csv(TOKENS_FILE).set_index("candidate_id")["tokens"].astype(float)
    a["tokens"] = t.reindex(a.index)
    return a
