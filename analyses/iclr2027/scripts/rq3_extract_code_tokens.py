"""Token counts of every candidate's code, from the raw run directories -> data/code_tokens.csv.gz.

Needs ROAR_RAW_DIR = the raw SkyDiscover run directories of the submitted corpus
(<run dir>/<framework>/<problem>/checkpoints/checkpoint_N/programs/<id>.json) and the
tiktoken o200k_base encoding (set TIKTOKEN_CACHE_DIR to a cache holding it for offline use).
Artifact uris in the mirror point at a transformed copy
(/ws/raw-data/gamble-data-transformed/<run dir>/output/checkpoints/<ck>/programs/<id>.json);
each is mapped to the raw file of the same checkpoint (or any checkpoint holding the same
program file) and used only if its SHA-256 equals artifacts.content_hash. Candidates of other
systems (CUDA, Dispatcher, KV-cache) have no raw text here and get no token count.
"""
import glob
import gzip
import hashlib
import os
import sys
import types
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from iclr2027 import code_length, paths  # noqa: E402

# export_for_embedding imports an LLM client at module level that is only used for its
# summary mode; stub it so the import has no side effects. Only load_code_from_uri is used.
_stub = types.ModuleType("litellm_client")
_stub.llm_batch_call_sync = None
sys.modules["litellm_client"] = _stub
import export_for_embedding as efe  # noqa: E402

PREFIX = "/ws/raw-data/gamble-data-transformed/"
RAW = paths.ROAR_RAW_DIR
_ENC = None


def _enc():
    global _ENC
    if _ENC is None:
        import tiktoken
        _ENC = tiktoken.get_encoding("o200k_base")
    return _ENC


def _local(uri):
    rest = uri[len(PREFIX):]
    rundir, _, tail = rest.partition("/")
    parts = tail.split("/")
    if len(parts) >= 4 and parts[0] == "output" and parts[1] == "checkpoints":
        ck, fname = parts[2], parts[-1]
        exact = glob.glob(f"{RAW}/{glob.escape(rundir)}/*/*/checkpoints/{ck}/programs/{fname}")
        return exact or sorted(glob.glob(f"{RAW}/{glob.escape(rundir)}/*/*/checkpoints/*/programs/{fname}"))
    return []


def _one(row):
    cid, uri, chash = row
    if not uri.startswith(PREFIX):
        return cid, None, "not_raw_skydiscover"
    for p in _local(uri):
        with open(p, "rb") as f:
            if chash is not None and hashlib.sha256(f.read()).hexdigest() != chash:
                continue
        code = efe.load_code_from_uri(p)
        if not code:
            return cid, None, "no_code"
        return cid, len(_enc().encode(code, disallowed_special=())), "ok"
    return cid, None, "missing_or_hash_mismatch"


def main():
    if not RAW or not os.path.isdir(RAW):
        sys.exit("set ROAR_RAW_DIR to the raw run directories")
    a = code_length.code_artifacts()
    _enc()
    with ProcessPoolExecutor(int(os.environ.get("WORKERS", "8"))) as ex:
        res = list(ex.map(_one, zip(a.candidate_id, a.uri, a.content_hash), chunksize=200))
    df = pd.DataFrame(res, columns=["candidate_id", "tokens", "status"])
    print(df.status.value_counts().to_dict())
    ok = df[df.status == "ok"][["candidate_id", "tokens"]].astype({"tokens": int}).sort_values("candidate_id")
    with gzip.GzipFile(paths.out(code_length.TOKENS_FILE), "wb", mtime=0) as f:
        f.write(ok.to_csv(index=False).encode())
    print("wrote", len(ok), "token counts")


if __name__ == "__main__":
    main()
