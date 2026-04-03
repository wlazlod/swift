#!/usr/bin/env bash
# =============================================================================
# run_all.sh — Orchestrate all SWIFT revision experiments
#
# Runs all 9 experiment scripts sequentially with progress tracking.
# Each experiment logs to a separate file under the specified log directory.
#
# Usage (from the project root: DWArticles/Codes/swift/):
#     bash experiments/run_all.sh              # Full run (~8 hours)
#     bash experiments/run_all.sh --fast       # Smoke test (~1 hour)
#
# Output:
#     results/v2/*.json         — Experiment result files
#     logs/<nn>_<name>.log      — Per-experiment log files
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RESULTS_DIR="$PROJECT_ROOT/results/v2"
LOG_DIR="$PROJECT_ROOT/logs"
VENV_DIR="$PROJECT_ROOT/.venv"

FAST_MODE=0
for arg in "$@"; do
    case "$arg" in
        --fast) FAST_MODE=1 ;;
        --help|-h)
            echo "Usage: bash experiments/run_all.sh [--fast]"
            echo ""
            echo "Options:"
            echo "  --fast    Run in fast mode (fewer reps/perms/seeds for smoke testing)"
            echo ""
            echo "Run from the project root: DWArticles/Codes/swift/"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: bash experiments/run_all.sh [--fast]"
            exit 1
            ;;
    esac
done

if [ "$FAST_MODE" -eq 1 ]; then
    FAST_FLAG="--fast"
    MODE_LABEL="FAST"
else
    FAST_FLAG=""
    MODE_LABEL="FULL"
fi

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

mkdir -p "$RESULTS_DIR"
mkdir -p "$LOG_DIR"

# Activate virtual environment
if [ -f "$VENV_DIR/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
else
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    echo "Create it first: uv venv && uv pip install -e '.[experiments]'"
    exit 1
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TOTAL_EXPERIMENTS=9
PASS_COUNT=0
FAIL_COUNT=0
OVERALL_START=$(date +%s)

# Arrays for the final summary
declare -a EXP_NAMES
declare -a EXP_STATUSES
declare -a EXP_DURATIONS

format_duration() {
    local secs=$1
    local mins=$((secs / 60))
    local remaining_secs=$((secs % 60))
    if [ "$mins" -gt 0 ]; then
        printf "%dm %ds" "$mins" "$remaining_secs"
    else
        printf "%ds" "$remaining_secs"
    fi
}

run_experiment() {
    local exp_num=$1
    local exp_name=$2
    local log_file="$LOG_DIR/$(printf '%02d' "$exp_num")_${exp_name}.log"
    shift 2
    local cmd=("$@")

    echo ""
    echo "=================================================================="
    echo "[$exp_num/$TOTAL_EXPERIMENTS] $exp_name"
    echo "=================================================================="
    echo "  Command : python -u ${cmd[*]}"
    echo "  Log     : $log_file"
    echo "  Started : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "------------------------------------------------------------------"

    local exp_start
    exp_start=$(date +%s)

    # Run with unbuffered output, tee to log file, preserve exit code
    set +e
    python -u "${cmd[@]}" 2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}
    set -e

    local exp_end
    exp_end=$(date +%s)
    local exp_elapsed=$((exp_end - exp_start))
    local duration_str
    duration_str=$(format_duration "$exp_elapsed")

    EXP_NAMES+=("$exp_name")
    EXP_DURATIONS+=("$duration_str")

    if [ "$exit_code" -ne 0 ]; then
        echo ""
        echo "  FAILED (exit code $exit_code) after $duration_str"
        echo "  See log: $log_file"
        echo ""
        echo "  Last 20 lines of log:"
        echo "  -----------------------"
        tail -20 "$log_file" | sed 's/^/  | /'
        EXP_STATUSES+=("FAILED")
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return 1
    fi

    echo ""
    echo "  PASSED in $duration_str"
    EXP_STATUSES+=("PASSED")
    PASS_COUNT=$((PASS_COUNT + 1))
    return 0
}

# ---------------------------------------------------------------------------
# Print header
# ---------------------------------------------------------------------------

echo "======================================================================"
echo "  SWIFT Experiment Runner"
echo "======================================================================"
echo "  Mode       : $MODE_LABEL"
echo "  Results    : $RESULTS_DIR"
echo "  Logs       : $LOG_DIR"
echo "  Python     : $(python --version 2>&1)"
echo "  Started    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"

# ---------------------------------------------------------------------------
# Experiment 1: Taiwan Credit — Controlled (S1-S9)
# ---------------------------------------------------------------------------

run_experiment 1 "taiwan_credit_controlled" \
    experiments/run_taiwan_credit.py \
    --output "$RESULTS_DIR/taiwan_credit_controlled.json" \
    $FAST_FLAG

# ---------------------------------------------------------------------------
# Experiment 2: Bank Marketing — Controlled (S1-S9)
# ---------------------------------------------------------------------------

run_experiment 2 "bank_marketing_controlled" \
    experiments/run_bank_marketing.py \
    --output "$RESULTS_DIR/bank_marketing_controlled.json" \
    $FAST_FLAG

# ---------------------------------------------------------------------------
# Experiment 3: Home Credit — Controlled (S1-S9)
# ---------------------------------------------------------------------------

run_experiment 3 "home_credit_controlled" \
    experiments/run_home_credit.py \
    --output "$RESULTS_DIR/home_credit_controlled.json" \
    $FAST_FLAG

# ---------------------------------------------------------------------------
# Experiment 4: Lending Club — Temporal Drift
# ---------------------------------------------------------------------------

run_experiment 4 "lending_club_temporal" \
    experiments/run_lending_club.py \
    --output "$RESULTS_DIR/lending_club_temporal.json" \
    $FAST_FLAG

# ---------------------------------------------------------------------------
# Experiment 5: Type I Error Calibration (all 3 datasets)
# ---------------------------------------------------------------------------

run_experiment 5 "calibration" \
    experiments/run_calibration.py \
    --output-dir "$RESULTS_DIR" \
    $FAST_FLAG

# ---------------------------------------------------------------------------
# Experiment 6: Power Analysis (Taiwan Credit)
# ---------------------------------------------------------------------------

run_experiment 6 "power_analysis" \
    experiments/run_power_analysis.py \
    --output "$RESULTS_DIR/power_analysis.json" \
    $FAST_FLAG

# ---------------------------------------------------------------------------
# Experiment 7: Multi-Seed Stability (all 3 datasets)
# ---------------------------------------------------------------------------

run_experiment 7 "multi_seed" \
    experiments/run_multi_seed.py \
    --output-dir "$RESULTS_DIR" \
    $FAST_FLAG

# ---------------------------------------------------------------------------
# Experiment 8: Gradual Drift S10 (all 3 datasets)
# ---------------------------------------------------------------------------

run_experiment 8 "gradual_drift" \
    experiments/run_gradual_drift.py \
    --output-dir "$RESULTS_DIR" \
    $FAST_FLAG

# ---------------------------------------------------------------------------
# Experiment 9: Ablations A1-A5 (all 3 datasets)
# ---------------------------------------------------------------------------

run_experiment 9 "ablations" \
    experiments/run_ablations.py \
    --output-dir "$RESULTS_DIR" \
    $FAST_FLAG

# ---------------------------------------------------------------------------
# Final Summary
# ---------------------------------------------------------------------------

OVERALL_END=$(date +%s)
OVERALL_ELAPSED=$((OVERALL_END - OVERALL_START))
OVERALL_DURATION=$(format_duration "$OVERALL_ELAPSED")

echo ""
echo ""
echo "======================================================================"
echo "  EXPERIMENT SUMMARY"
echo "======================================================================"
printf "  %-4s %-30s %-8s %s\n" "#" "Experiment" "Status" "Duration"
echo "  ---- ------------------------------ -------- ----------"
for i in "${!EXP_NAMES[@]}"; do
    local_num=$((i + 1))
    printf "  %-4s %-30s %-8s %s\n" \
        "$local_num" "${EXP_NAMES[$i]}" "${EXP_STATUSES[$i]}" "${EXP_DURATIONS[$i]}"
done
echo "  ---- ------------------------------ -------- ----------"
echo ""
echo "  Total     : $OVERALL_DURATION"
echo "  Passed    : $PASS_COUNT / $TOTAL_EXPERIMENTS"
echo "  Failed    : $FAIL_COUNT / $TOTAL_EXPERIMENTS"
echo "  Mode      : $MODE_LABEL"
echo "  Results   : $RESULTS_DIR"
echo "  Logs      : $LOG_DIR"
echo "  Finished  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"

if [ "$FAIL_COUNT" -gt 0 ]; then
    echo ""
    echo "WARNING: $FAIL_COUNT experiment(s) failed. Check logs for details."
    exit 1
fi

echo ""
echo "All experiments completed successfully!"
echo ""
echo "Next steps:"
echo "  1. Review results in $RESULTS_DIR"
echo "  2. Generate figures and tables:"
echo "     python scripts/analyze_results.py"
echo "  3. Integrate into paper (Phase 2)"
