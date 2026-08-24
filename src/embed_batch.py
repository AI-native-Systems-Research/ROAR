#!/usr/bin/env python3
"""Batch embedding script for GPU cluster.

Reads JSONL with {candidate_id, text}, outputs JSONL with {candidate_id, embedding}.
No database access needed - designed to run on GPU cluster with minimal dependencies.

Models:
  --model text   BAAI/bge-small-en-v1.5 (384-dim) — for summaries
  --model code   jinaai/jina-embeddings-v2-base-code (768-dim) — for raw code

Chunking (for long code files):
  --chunk-size N    Split text into chunks of ~N tokens (default: no chunking)
  --chunk-overlap M Overlap between chunks in tokens (default: 128)
  --max-tokens N    Max tokens per GPU batch (default: 16384) — controls memory usage
  Chunk embeddings are averaged to produce final embedding per candidate.

GPU acceleration:
  Set FASTEMBED_GPU=1 to enable CUDA (requires fastembed-gpu package).

Usage:
    python embed_batch.py --input summaries.jsonl --output summary_embeddings.jsonl --model text
    python embed_batch.py --input code_to_embed.jsonl --output code_embeddings.jsonl --model code
    python embed_batch.py --input code.jsonl --output emb.jsonl --model code --chunk-size 2048 --max-tokens 16384

LSF job example:
    #BSUB -q gpu
    #BSUB -gpu "num=1:gmem=70G"
    #BSUB -o embed_%J.out
    python embed_batch.py --input code.jsonl --output emb.jsonl --model code --chunk-size 2048 --max-tokens 16384
"""

import argparse
import json
import sys

from tqdm import tqdm

from embeddings import EmbeddingModel, embed_texts, get_dimensions

CHARS_PER_TOKEN = 4
DEFAULT_MAX_TOKENS = 65536  # Total tokens per batch - controls GPU memory


def count_lines(path: str) -> int:
    """Count lines in a file."""
    count = 0
    with open(path) as f:
        for _ in f:
            count += 1
    return count


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks based on approximate token count.

    Args:
        text: The text to chunk
        chunk_size: Target chunk size in tokens
        overlap: Overlap between chunks in tokens

    Returns:
        List of text chunks
    """
    chunk_chars = chunk_size * CHARS_PER_TOKEN
    overlap_chars = overlap * CHARS_PER_TOKEN
    step = chunk_chars - overlap_chars

    if step <= 0:
        step = chunk_chars // 2

    if len(text) <= chunk_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_chars
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += step

    return chunks if chunks else [text]


def count_chunks(text: str, chunk_size: int, overlap: int) -> int:
    """Count how many chunks a text will produce."""
    chunk_chars = chunk_size * CHARS_PER_TOKEN
    overlap_chars = overlap * CHARS_PER_TOKEN
    step = chunk_chars - overlap_chars

    if step <= 0:
        step = chunk_chars // 2

    if len(text) <= chunk_chars:
        return 1

    return max(1, (len(text) - overlap_chars + step - 1) // step)


def load_input(path: str) -> list[dict]:
    """Load input JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def average_embeddings(embeddings: list[list[float]]) -> list[float]:
    """Average multiple embeddings into one."""
    if len(embeddings) == 1:
        return embeddings[0]

    dim = len(embeddings[0])
    result = [0.0] * dim
    for emb in embeddings:
        for i, v in enumerate(emb):
            result[i] += v
    return [v / len(embeddings) for v in result]


def load_completed_ids(output_path: str) -> set[str]:
    """Load candidate IDs that have already been embedded."""
    completed = set()
    try:
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    completed.add(record["candidate_id"])
    except FileNotFoundError:
        pass
    return completed


def embed_batch_file(
    input_path: str,
    output_path: str,
    model: EmbeddingModel,
    chunk_size: int | None = None,
    chunk_overlap: int = 128,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    resume: bool = False,
) -> int:
    """Embed all texts in input file, write to output file.

    If chunk_size is set, long texts are split into overlapping chunks,
    embedded separately, and averaged to produce the final embedding.

    Records are sorted by length and batched by total chunk count (max_chunks)
    to maximize GPU efficiency and prevent OOM.

    If resume=True, skip records already in output file and append new results.
    """
    print(f"Loading {input_path}...", flush=True)
    records = load_input(input_path)
    total = len(records)

    if total == 0:
        print("No records to embed.", flush=True)
        return 0

    print(f"Loaded {total} records.", flush=True)
    print(f"Model: {model.value} ({get_dimensions(model)}-dim)", flush=True)

    # Pre-compute token counts and sort by text length
    if chunk_size:
        print(f"Chunking: {chunk_size} tokens with {chunk_overlap} token overlap", flush=True)
        print(f"Max tokens per batch: {max_tokens}", flush=True)

        for record in records:
            record["_num_chunks"] = count_chunks(record["text"], chunk_size, chunk_overlap)
            record["_text_len"] = len(record["text"])
            record["_num_tokens"] = record["_text_len"] // CHARS_PER_TOKEN

        # Sort by text length (shortest first) for efficient packing
        records.sort(key=lambda r: r["_text_len"])

        total_chunks = sum(r["_num_chunks"] for r in records)
        avg_chunks = total_chunks / total
        print(f"Total chunks: {total_chunks} (avg {avg_chunks:.1f} per record)", flush=True)

    # Handle resume mode
    completed_ids: set[str] = set()
    if resume:
        completed_ids = load_completed_ids(output_path)
        if completed_ids:
            print(f"Resume mode: {len(completed_ids)} already completed, skipping", flush=True)
            records = [r for r in records if r["candidate_id"] not in completed_ids]
            total = len(records)
            if total == 0:
                print("All records already completed.", flush=True)
                return 0
            print(f"Remaining: {total} records", flush=True)

    # Check ONNX execution providers
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"ONNX providers available: {providers}", flush=True)
        if "CUDAExecutionProvider" in providers:
            print("GPU acceleration: ENABLED", flush=True)
        else:
            print("WARNING: GPU acceleration NOT available - running on CPU!", flush=True)
    except Exception as e:
        print(f"Could not check ONNX providers: {e}", flush=True)

    # Open in append mode if resuming, write mode otherwise
    file_mode = "a" if resume and completed_ids else "w"
    with open(output_path, file_mode) as f:
        pbar = tqdm(total=total, desc="Embedding", unit="rec")

        if chunk_size:
            # Token-limited batching for consistent GPU memory usage
            idx = 0
            while idx < total:
                # Accumulate records until we hit max_tokens
                batch = []
                batch_token_count = 0

                while idx < total:
                    record = records[idx]
                    record_tokens = record["_num_tokens"]

                    # Always include at least one record per batch
                    if batch and batch_token_count + record_tokens > max_tokens:
                        break

                    batch.append(record)
                    batch_token_count += record_tokens
                    idx += 1

                # Process this batch
                all_chunks: list[str] = []
                chunk_map: list[tuple[int, int]] = []  # (batch_idx, num_chunks)

                for i, record in enumerate(batch):
                    chunks = chunk_text(record["text"], chunk_size, chunk_overlap)
                    chunk_map.append((i, len(chunks)))
                    all_chunks.extend(chunks)

                # Embed all chunks at once
                all_embeddings = embed_texts(all_chunks, model=model)

                # Reassemble: average chunks per record
                chunk_idx = 0
                for batch_idx, num_chunks in chunk_map:
                    record_chunks = all_embeddings[chunk_idx:chunk_idx + num_chunks]
                    chunk_idx += num_chunks
                    avg_embedding = average_embeddings(record_chunks)

                    output_record = {
                        "candidate_id": batch[batch_idx]["candidate_id"],
                        "embedding": avg_embedding,
                    }
                    f.write(json.dumps(output_record) + "\n")

                pbar.update(len(batch))
        else:
            # No chunking - process all at once
            texts = [r["text"] for r in records]
            embeddings = embed_texts(texts, model=model)

            for record, embedding in zip(records, embeddings):
                output_record = {
                    "candidate_id": record["candidate_id"],
                    "embedding": embedding,
                }
                f.write(json.dumps(output_record) + "\n")

            pbar.update(total)

        pbar.close()

    return total


def main():
    parser = argparse.ArgumentParser(
        description="Batch embedding for GPU cluster."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input JSONL file with {candidate_id, text}",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output JSONL file with {candidate_id, embedding}",
    )
    parser.add_argument(
        "--model",
        choices=["text", "code"],
        required=True,
        help="text: bge-small-en (384-dim). code: jina-code-v2 (768-dim).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Split long texts into chunks of ~N tokens (default: no chunking)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=128,
        help="Overlap between chunks in tokens (default: 128)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Max tokens per GPU batch (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file, skipping already embedded records",
    )

    args = parser.parse_args()

    model = EmbeddingModel.TEXT if args.model == "text" else EmbeddingModel.CODE

    try:
        embed_batch_file(
            input_path=args.input,
            output_path=args.output,
            model=model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            max_tokens=args.max_tokens,
            resume=args.resume,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSONL: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
