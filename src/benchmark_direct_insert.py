#!/usr/bin/env python3
"""Benchmark direct insertion of gamble-data-transformed without tar/untar.

Measures:
- Parse time per campaign
- Database insert time per campaign
- Total time
- Data counts
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from insert_adrs_campaign import CampaignInserter, detect_system_type, _parse_single_campaign


def get_folder_stats(folder_path: Path) -> dict:
    """Get size and file counts for a folder."""
    total_size = 0
    file_count = 0
    json_count = 0

    for root, _dirs, files in os.walk(folder_path):
        for f in files:
            fp = Path(root) / f
            try:
                total_size += fp.stat().st_size
                file_count += 1
                if f.endswith('.json'):
                    json_count += 1
            except OSError:
                pass

    return {
        'total_size_bytes': total_size,
        'total_size_mb': total_size / (1024 * 1024),
        'file_count': file_count,
        'json_count': json_count,
    }


def benchmark_insert(campaigns_dir: Path, max_workers: int = 8) -> dict:
    """Benchmark parsing and insertion of all campaigns."""

    results = {
        'campaigns_dir': str(campaigns_dir),
        'campaigns': [],
        'parse_times': [],
        'insert_times': [],
        'errors': [],
    }

    # Collect campaign folders
    campaign_folders = []
    for item in sorted(campaigns_dir.iterdir()):
        if item.is_dir() and detect_system_type(item) is not None:
            campaign_folders.append(item)

    results['campaign_count'] = len(campaign_folders)
    print(f"Found {len(campaign_folders)} campaigns to process")

    # Get overall folder stats
    print("Calculating folder statistics...")
    stats_start = time.time()
    folder_stats = get_folder_stats(campaigns_dir)
    stats_time = time.time() - stats_start
    results['folder_stats'] = folder_stats
    results['stats_calc_time'] = stats_time
    print(f"  Size: {folder_stats['total_size_mb']:.2f} MB")
    print(f"  Files: {folder_stats['file_count']}")
    print(f"  JSON files: {folder_stats['json_count']}")
    print(f"  Stats calc time: {stats_time:.2f}s")

    # Phase 1: Parse all campaigns in parallel
    print("\nPhase 1: Parsing campaigns...")
    parse_start = time.time()
    parsed_campaigns = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for folder in campaign_folders:
            future = executor.submit(_parse_single_campaign, folder, 'skydiscover')
            futures[future] = (folder, time.time())

        for i, future in enumerate(as_completed(futures)):
            folder, submit_time = futures[future]
            path, parsed, error = future.result()
            parse_time = time.time() - submit_time

            if parsed:
                parsed_campaigns.append((path, parsed, parse_time))
                results['parse_times'].append(parse_time)
                if (i + 1) % 100 == 0:
                    print(f"  Parsed {i + 1}/{len(campaign_folders)} campaigns...")
            else:
                results['errors'].append({
                    'campaign': path.name,
                    'phase': 'parse',
                    'error': error
                })

    total_parse_time = time.time() - parse_start
    results['total_parse_time'] = total_parse_time
    print(f"  Total parse time: {total_parse_time:.2f}s")
    print(f"  Successfully parsed: {len(parsed_campaigns)}/{len(campaign_folders)}")

    # Collect candidate/measurement counts
    total_candidates = 0
    total_measurements = 0
    total_artifacts = 0
    total_edges = 0

    for _, parsed, _ in parsed_campaigns:
        total_candidates += len(parsed.candidates)
        total_measurements += sum(len(m) for m in parsed.measurements.values())
        total_artifacts += len(parsed.artifacts)
        total_edges += len(parsed.candidate_edges)

    results['total_candidates'] = total_candidates
    results['total_measurements'] = total_measurements
    results['total_artifacts'] = total_artifacts
    results['total_edges'] = total_edges

    print(f"  Total candidates: {total_candidates}")
    print(f"  Total measurements: {total_measurements}")
    print(f"  Total artifacts: {total_artifacts}")
    print(f"  Total edges: {total_edges}")

    # Phase 2: Insert all campaigns
    print("\nPhase 2: Inserting campaigns into database...")
    insert_start = time.time()

    inserter = CampaignInserter()
    success_count = 0

    for i, (path, parsed, _parse_time) in enumerate(parsed_campaigns):
        campaign_insert_start = time.time()
        try:
            if inserter.insert(parsed):
                insert_time = time.time() - campaign_insert_start
                results['insert_times'].append(insert_time)
                success_count += 1

                if (i + 1) % 100 == 0:
                    print(f"  Inserted {i + 1}/{len(parsed_campaigns)} campaigns...")
            else:
                results['errors'].append({
                    'campaign': path.name,
                    'phase': 'insert',
                    'error': 'insert returned False'
                })
        except Exception as e:
            results['errors'].append({
                'campaign': path.name,
                'phase': 'insert',
                'error': str(e)
            })
        finally:
            inserter.system_id = None
            inserter.campaign_id = None
            inserter.candidate_ids = {}

    total_insert_time = time.time() - insert_start
    results['total_insert_time'] = total_insert_time
    results['success_count'] = success_count
    results['failure_count'] = len(campaign_folders) - success_count

    print(f"  Total insert time: {total_insert_time:.2f}s")
    print(f"  Successfully inserted: {success_count}/{len(parsed_campaigns)}")

    # Summary
    results['total_time'] = total_parse_time + total_insert_time

    return results


def write_report(results: dict, output_path: Path):
    """Write benchmark results to markdown file."""

    with open(output_path, 'w') as f:
        f.write("# Gamble-Data-Transformed Direct Upload Benchmark\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Data Summary\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Source Directory | `{results['campaigns_dir']}` |\n")
        f.write(f"| Total Size | {results['folder_stats']['total_size_mb']:.2f} MB |\n")
        f.write(f"| Total Files | {results['folder_stats']['file_count']:,} |\n")
        f.write(f"| JSON Files | {results['folder_stats']['json_count']:,} |\n")
        f.write(f"| Campaigns | {results['campaign_count']} |\n")
        f.write(f"| Candidates | {results['total_candidates']:,} |\n")
        f.write(f"| Measurements | {results['total_measurements']:,} |\n")
        f.write(f"| Artifacts | {results['total_artifacts']:,} |\n")
        f.write(f"| Edges | {results['total_edges']:,} |\n")

        f.write("\n## Timing Results\n\n")
        f.write(f"| Phase | Time (s) | Notes |\n")
        f.write(f"|-------|----------|-------|\n")
        f.write(f"| Tar Creation | N/A | Direct insert, no tar |\n")
        f.write(f"| Upload | N/A | Direct insert, no upload |\n")
        f.write(f"| Extraction | N/A | Direct insert, no extraction |\n")
        f.write(f"| Parse (all campaigns) | {results['total_parse_time']:.2f} | Parallel parsing |\n")
        f.write(f"| Database Insert | {results['total_insert_time']:.2f} | Sequential inserts |\n")
        f.write(f"| **Total** | **{results['total_time']:.2f}** | |\n")

        f.write("\n## Performance Metrics\n\n")

        if results['parse_times']:
            avg_parse = sum(results['parse_times']) / len(results['parse_times'])
            f.write(f"- Average parse time per campaign: {avg_parse:.3f}s\n")

        if results['insert_times']:
            avg_insert = sum(results['insert_times']) / len(results['insert_times'])
            f.write(f"- Average insert time per campaign: {avg_insert:.3f}s\n")

        if results['total_time'] > 0:
            throughput_mb = results['folder_stats']['total_size_mb'] / results['total_time']
            throughput_campaigns = results['success_count'] / results['total_time']
            f.write(f"- Throughput: {throughput_mb:.2f} MB/s\n")
            f.write(f"- Throughput: {throughput_campaigns:.2f} campaigns/s\n")

        f.write(f"\n## Results\n\n")
        f.write(f"- Successfully processed: {results['success_count']}/{results['campaign_count']} campaigns\n")
        f.write(f"- Failed: {results['failure_count']} campaigns\n")

        if results['errors']:
            f.write(f"\n### Errors ({len(results['errors'])})\n\n")
            for err in results['errors'][:20]:  # Limit to first 20
                f.write(f"- `{err['campaign']}` ({err['phase']}): {err['error']}\n")
            if len(results['errors']) > 20:
                f.write(f"- ... and {len(results['errors']) - 20} more errors\n")

    print(f"\nReport written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark direct insertion of campaigns")
    parser.add_argument("path", type=Path, help="Path to campaigns directory")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers for parsing")
    parser.add_argument("--output", type=Path, default=Path("/ws/gamble_upload_benchmark.md"),
                       help="Output markdown file")
    args = parser.parse_args()

    print(f"Starting benchmark of {args.path}")
    print(f"Workers: {args.workers}")
    print()

    results = benchmark_insert(args.path, max_workers=args.workers)
    write_report(results, args.output)

    print(f"\n{'='*60}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {results['total_time']:.2f}s")
    print(f"Campaigns: {results['success_count']}/{results['campaign_count']}")
    print(f"Candidates: {results['total_candidates']:,}")


if __name__ == "__main__":
    main()
