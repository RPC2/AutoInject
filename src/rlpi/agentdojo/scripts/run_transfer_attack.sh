#!/bin/bash

# Function to run a single transfer attack
run_transfer_attack() {
    local source_model=${1:-google/gemini-2.0-flash-001}
    local target_model=${2:-google/gemini-2.0-flash-001}
    local test_tasks=${3:-banking}
    # Sanitize model names for directory names (replace / with _)
    local source_safe=$(echo ${source_model} | tr '/' '_')
    local target_safe=$(echo ${target_model} | tr '/' '_')

    echo "=========================================="
    echo "Running transfer attack:"
    echo "  Source: ${source_model}"
    echo "  Target: ${target_model}"
    echo "=========================================="

    python -m rlpi.agentdojo.transfer_attack \
        source_model=${source_model} \
        target_model=${target_model} \
        test_tasks=${test_tasks} \
        model_attack_strings=${test_tasks} \
        exp_ident=transfer_${source_safe}_to_${target_safe} \
        "hydra.run.dir=./outputs/transfer/\${now:%Y-%m-%d}/sequential-\${now:%H-%M-%S}/${source_safe}_to_${target_safe}"

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✓ Completed: ${source_model} -> ${target_model}"
    else
        echo "✗ Failed: ${source_model} -> ${target_model} (exit code: $exit_code)"
    fi
    echo ""
}

# Run all transfer attack combinations sequentially
source_models=(gpt-4.1-nano gpt-4o-mini-2024-07-18 gpt-5-nano google/gemini-2.0-flash-001 google/gemini-2.5-flash)
target_models=(facebook/Meta-SecAlign-70B)
test_tasks=(slack)

total_combinations=$((${#source_models[@]} * ${#target_models[@]}))
current=0

echo "Starting sequential transfer attack runs"
echo "Total combinations: ${total_combinations}"
echo "Results will be saved in: ./outputs/transfer/sequential/"
echo ""

for source_model in "${source_models[@]}"; do
    for target_model in "${target_models[@]}"; do
        current=$((current + 1))
        echo "[${current}/${total_combinations}]"
        run_transfer_attack "${source_model}" "${target_model}" "${test_tasks}"
    done
done

echo "=========================================="
echo "All transfer attack runs completed!"
echo "Results saved in: ./outputs/transfer/sequential/"
echo "=========================================="
