#!/usr/bin/env bash
set -Eeuo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_dir="${1:-${root_dir}/evaluation_runs/baselines_$(date +%Y%m%d_%H%M%S)}"
python_bin="${PYTHON_BIN:-python}"
poll_seconds="${GPU_POLL_SECONDS:-30}"
max_memory_mib="${GPU_IDLE_MAX_MEMORY_MIB:-500}"
max_utilization="${GPU_IDLE_MAX_UTILIZATION:-5}"
raft_checkpoint="${root_dir}/.cache/vbench/raft_model/models/raft-things.pth"

mkdir -p "${run_dir}/logs"
exec > >(tee -a "${run_dir}/logs/orchestrator.log") 2>&1
printf '[%s] StoryMem/ViMax evaluation run: %s\n' "$(date --iso-8601=seconds)" "${run_dir}"

[[ -x "${python_bin}" ]] || { echo "missing Python: ${python_bin}" >&2; exit 2; }
[[ -f "${raft_checkpoint}" ]] || { echo "missing RAFT checkpoint: ${raft_checkpoint}" >&2; exit 2; }
"${python_bin}" "${root_dir}/tools/prepare_baseline_run.py" "${run_dir}"

gpu=""
lock_dir=""
while [[ -z "${gpu}" ]]; do
    while IFS=',' read -r raw_index raw_memory raw_utilization; do
        index="${raw_index// /}"
        memory="${raw_memory// /}"
        utilization="${raw_utilization// /}"
        [[ "${memory}" =~ ^[0-9]+$ && "${utilization}" =~ ^[0-9]+$ ]] || continue
        if (( memory <= max_memory_mib && utilization <= max_utilization )); then
            candidate_lock="/tmp/vbench_baselines_gpu_${index}.lock"
            if mkdir "${candidate_lock}" 2>/dev/null; then
                gpu="${index}"
                lock_dir="${candidate_lock}"
                printf '%s\n' "$$" > "${lock_dir}/pid"
                break
            fi
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
    if [[ -z "${gpu}" ]]; then
        printf '[%s] no idle GPU (limits: memory<=%s MiB, utilization<=%s%%); retrying in %ss\n' \
            "$(date --iso-8601=seconds)" "${max_memory_mib}" "${max_utilization}" "${poll_seconds}"
        sleep "${poll_seconds}"
    fi
done
trap 'rm -f "${lock_dir}/pid"; rmdir "${lock_dir}" 2>/dev/null || true' EXIT
printf '[%s] acquired GPU %s\n' "$(date --iso-8601=seconds)" "${gpu}"

export PATH="$(dirname "${python_bin}"):${PATH}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export VBENCH_CACHE_DIR="${root_dir}/.cache/vbench"
export HF_HOME="${root_dir}/.cache/huggingface"
export TORCH_HOME="${root_dir}/.cache/torch"
export PYTHONPATH="${root_dir}/vbench/third_party/RAFT/alt_cuda_corr:${root_dir}:${PYTHONPATH:-}"

cases=(dualjustice1 dualjustice2 elf2 elf3 goblin kick lovehina mado)
methods=(storymem vimax manga2animate)
metrics=(i2v_subject i2v_background subject_consistency background_consistency motion_smoothness dynamic_degree aesthetic_quality imaging_quality)

for method in "${methods[@]}"; do
    method_cases=("${cases[@]}")
    if [[ "${method}" == "manga2animate" ]]; then
        method_cases=(dualjustice2)
    fi
    for case_name in "${method_cases[@]}"; do
        for metric in "${metrics[@]}"; do
            "${python_bin}" "${root_dir}/tools/run_baseline_metric.py" \
                --run-dir "${run_dir}" --case "${case_name}" --method "${method}" --metric "${metric}" --gpu "${gpu}"
        done
    done
    "${python_bin}" "${root_dir}/tools/modified_dynamic_degree.py" \
        --manifest "${run_dir}/audit/${method}_manifest.json" \
        --output "${run_dir}/raw/modified_dd_${method}.csv" \
        --checkpoint "${raft_checkpoint}" --device cuda:0 --alternate-corr
    "${python_bin}" "${root_dir}/tools/evaluate_i2v_subject_dinov3_official.py" \
        --manifest "${run_dir}/audit/dinov3_manifest.json" \
        --output "${run_dir}/raw/dinov3_${method}.csv" --method "${method}" --gpu "${gpu}"
done

"${python_bin}" "${root_dir}/tools/aggregate_baseline_run.py" "${run_dir}"
printf '[%s] StoryMem/ViMax evaluation completed successfully\n' "$(date --iso-8601=seconds)"
