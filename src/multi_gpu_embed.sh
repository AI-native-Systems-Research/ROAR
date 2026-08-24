#!/bin/bash
#==============================================================================
# Multi-GPU Embedding Helper Script
#==============================================================================
#
# Splits input file, submits parallel GPU jobs, and merges results.
#
# Usage:
#   ./multi_gpu_embed.sh split <input_file> <num_parts>
#   ./multi_gpu_embed.sh submit
#   ./multi_gpu_embed.sh status
#   ./multi_gpu_embed.sh merge <output_file>
#   ./multi_gpu_embed.sh clean
#
# Workflow:
#   1. ./multi_gpu_embed.sh split direct_code_to_embed.jsonl 4
#   2. ./multi_gpu_embed.sh submit
#   3. ./multi_gpu_embed.sh status   # repeat until all done
#   4. ./multi_gpu_embed.sh merge direct_code_embeddings.jsonl
#   5. ./multi_gpu_embed.sh clean    # remove temp files
#
#==============================================================================

set -e

# Directory for split files and job scripts
WORK_DIR="multi_gpu_work"
JOB_IDS_FILE="$WORK_DIR/job_ids.txt"

#------------------------------------------------------------------------------
# SPLIT: Split input file into N parts
#------------------------------------------------------------------------------
do_split() {
    local input_file="$1"
    local num_parts="$2"

    if [[ -z "$input_file" || -z "$num_parts" ]]; then
        echo "Usage: $0 split <input_file> <num_parts>"
        exit 1
    fi

    if [[ ! -f "$input_file" ]]; then
        echo "Error: Input file not found: $input_file"
        exit 1
    fi

    mkdir -p "$WORK_DIR"

    echo "Splitting $input_file into $num_parts parts..."
    total_lines=$(wc -l < "$input_file")
    echo "Total records: $total_lines"

    # Split into N parts (by line count, not bytes)
    split -n l/"$num_parts" "$input_file" "$WORK_DIR/part_"

    # List created parts
    echo ""
    echo "Created parts:"
    for part in "$WORK_DIR"/part_*; do
        lines=$(wc -l < "$part")
        echo "  $part: $lines records"
    done

    # Save config for later steps
    echo "$input_file" > "$WORK_DIR/config.txt"
    echo "$num_parts" >> "$WORK_DIR/config.txt"

    echo ""
    echo "Next step: ./multi_gpu_embed.sh submit"
}

#------------------------------------------------------------------------------
# SUBMIT: Create job scripts and submit all parts
#------------------------------------------------------------------------------
do_submit() {
    if [[ ! -d "$WORK_DIR" ]]; then
        echo "Error: Run 'split' first"
        exit 1
    fi

    # Clear previous job IDs
    > "$JOB_IDS_FILE"

    echo "Submitting jobs..."
    echo ""

    for part in "$WORK_DIR"/part_*; do
        part_name=$(basename "$part")
        output_file="${part}_embeddings.jsonl"
        job_script="${part}.lsf"

        # Create job script for this part
        cat > "$job_script" << 'HEADER'
#!/bin/bash
#BSUB -q normal
#BSUB -gpu "num=1:gmem=80G"
#BSUB -n 4
#BSUB -R "rusage[mem=64GB]"
#BSUB -W 1:00
HEADER

        cat >> "$job_script" << EOF
#BSUB -J embed_${part_name}
#BSUB -o ${part}_job_%J.out

export CUDA_HOME=/opt/share/cuda-12.4
export CUDNN_HOME=/opt/share/cudnn-linux-x86_64-9.8.0.87_cuda12-archive
export PATH="\$CUDA_HOME/bin:\$PATH"
export LD_LIBRARY_PATH="\$CUDA_HOME/lib64:\$CUDNN_HOME/lib:\$LD_LIBRARY_PATH"
export FASTEMBED_GPU=1

cd ~/adrs_data/src

echo "Processing: $part"
echo "Output: $output_file"
echo "Host: \$(hostname)"
echo "Start: \$(date)"

uv run python embed_batch.py \\
    --input "$part" \\
    --output "$output_file" \\
    --model code \\
    --chunk-size 2048 \\
    --chunk-overlap 256 \\
    --max-tokens 32768 \\
    --resume

echo "Finished: \$(date)"
echo "Output lines: \$(wc -l < "$output_file")"
EOF

        # Submit and capture job ID
        job_output=$(bsub < "$job_script" 2>&1)
        job_id=$(echo "$job_output" | grep -oP '(?<=Job <)\d+(?=>)')

        if [[ -n "$job_id" ]]; then
            echo "$job_id" >> "$JOB_IDS_FILE"
            echo "  Submitted $part_name -> Job $job_id"
        else
            echo "  Warning: Failed to submit $part_name"
            echo "  $job_output"
        fi
    done

    echo ""
    echo "All jobs submitted. Job IDs saved to $JOB_IDS_FILE"
    echo ""
    echo "Next step: ./multi_gpu_embed.sh status"
}

#------------------------------------------------------------------------------
# STATUS: Check status of all submitted jobs
#------------------------------------------------------------------------------
do_status() {
    if [[ ! -f "$JOB_IDS_FILE" ]]; then
        echo "Error: No jobs found. Run 'submit' first."
        exit 1
    fi

    echo "Job Status:"
    echo "==========="

    local all_done=true
    local completed=0
    local running=0
    local pending=0
    local failed=0

    while read -r job_id; do
        status=$(bjobs -o "stat" -noheader "$job_id" 2>/dev/null || echo "DONE")

        case "$status" in
            RUN)
                echo "  Job $job_id: RUNNING"
                running=$((running + 1))
                all_done=false
                ;;
            PEND)
                echo "  Job $job_id: PENDING"
                pending=$((pending + 1))
                all_done=false
                ;;
            DONE)
                echo "  Job $job_id: COMPLETED"
                completed=$((completed + 1))
                ;;
            EXIT)
                echo "  Job $job_id: FAILED"
                failed=$((failed + 1))
                ;;
            *)
                echo "  Job $job_id: $status"
                ;;
        esac
    done < "$JOB_IDS_FILE"

    echo ""
    echo "Summary: $completed completed, $running running, $pending pending, $failed failed"

    # Check output files
    echo ""
    echo "Output Files:"
    for part in "$WORK_DIR"/part_*_embeddings.jsonl 2>/dev/null; do
        if [[ -f "$part" ]]; then
            lines=$(wc -l < "$part")
            echo "  $part: $lines records"
        fi
    done

    if $all_done && [[ $failed -eq 0 ]]; then
        echo ""
        echo "All jobs completed successfully!"
        echo "Next step: ./multi_gpu_embed.sh merge <output_file>"
    elif [[ $failed -gt 0 ]]; then
        echo ""
        echo "Some jobs failed. Check the .out files for errors."
    fi
}

#------------------------------------------------------------------------------
# MERGE: Combine all output files
#------------------------------------------------------------------------------
do_merge() {
    local output_file="$1"

    if [[ -z "$output_file" ]]; then
        echo "Usage: $0 merge <output_file>"
        exit 1
    fi

    # Find all embedding output files
    local parts=("$WORK_DIR"/part_*_embeddings.jsonl)

    if [[ ${#parts[@]} -eq 0 || ! -f "${parts[0]}" ]]; then
        echo "Error: No output files found in $WORK_DIR"
        exit 1
    fi

    echo "Merging ${#parts[@]} files into $output_file..."

    # Concatenate all parts
    cat "${parts[@]}" > "$output_file"

    total_lines=$(wc -l < "$output_file")
    echo "Total records: $total_lines"

    # Verify no duplicates
    unique_ids=$(cut -d'"' -f4 "$output_file" | sort -u | wc -l)
    echo "Unique candidate_ids: $unique_ids"

    if [[ "$total_lines" -ne "$unique_ids" ]]; then
        echo "Warning: Found duplicate candidate_ids!"
    else
        echo "Verification passed: no duplicates"
    fi

    echo ""
    echo "Output written to: $output_file"
    echo "Next step: ./multi_gpu_embed.sh clean (optional)"
}

#------------------------------------------------------------------------------
# CLEAN: Remove temporary files
#------------------------------------------------------------------------------
do_clean() {
    if [[ ! -d "$WORK_DIR" ]]; then
        echo "Nothing to clean"
        exit 0
    fi

    echo "Files to remove:"
    ls -la "$WORK_DIR"/ 2>/dev/null || true

    echo ""
    read -p "Remove all files in $WORK_DIR? [y/N] " confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        rm -rf "$WORK_DIR"
        echo "Cleaned up $WORK_DIR"
    else
        echo "Cancelled"
    fi
}

#------------------------------------------------------------------------------
# MAIN
#------------------------------------------------------------------------------
case "${1:-}" in
    split)
        do_split "$2" "$3"
        ;;
    submit)
        do_submit
        ;;
    status)
        do_status
        ;;
    merge)
        do_merge "$2"
        ;;
    clean)
        do_clean
        ;;
    *)
        echo "Multi-GPU Embedding Helper"
        echo ""
        echo "Usage:"
        echo "  $0 split <input_file> <num_parts>  - Split input into N parts"
        echo "  $0 submit                          - Submit all jobs"
        echo "  $0 status                          - Check job status"
        echo "  $0 merge <output_file>             - Merge all outputs"
        echo "  $0 clean                           - Remove temp files"
        echo ""
        echo "Example workflow:"
        echo "  $0 split direct_code_to_embed.jsonl 4"
        echo "  $0 submit"
        echo "  $0 status"
        echo "  $0 merge direct_code_embeddings.jsonl"
        echo "  $0 clean"
        ;;
esac
