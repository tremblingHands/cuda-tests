#!/bin/bash

# GPU 与 NUMA 映射关系
declare -A GPU_NUMA_MAP
GPU_NUMA_MAP[0]=0
GPU_NUMA_MAP[1]=0
GPU_NUMA_MAP[2]=1
GPU_NUMA_MAP[3]=1

# 测试脚本路径
D2H_PATH="/home/nathan/cuda-tests/d2h"
H2D_PATH="/home/nathan/cuda-tests/h2d"

# 测试配置
DATA_SIZES=(1 10 20 40)
ITERATIONS=1

# 单实例测试GPU
SINGLE_GPU_ID=0

# 多实例测试GPU组合
MULTI_2GPU_INSTANCES=(
    "0:local 1:local"
    "0:local 2:remote"
    "0:remote 2:remote"
)

MULTI_3GPU_INSTANCES=(
    "0:local 1:local 2:remote"
    "0:remote 1:remote 2:local"
    "0:remote 1:remote 2:remote"
)

MULTI_4GPU_INSTANCES=(
    "0:local 1:local 2:local 3:local"
    "0:remote 1:remote 2:remote 3:remote"
    "0:local 1:remote 2:local 3:remote"
)

# 单GPU评估函数
eval_single_gpu() {
    local gpu_id=$1
    local mode=$2
    local test_type=$3
    local data_size=$4
    local iterations=$5

    local numa_node=${GPU_NUMA_MAP[$gpu_id]}
    local mem_node
    if [ "$mode" == "local" ]; then
        mem_node=$numa_node
    elif [ "$mode" == "remote" ]; then
        mem_node=$((1 - numa_node))
    else
        echo "Error: mode must be 'local' or 'remote'"
        return 1
    fi

    local test_path
    if [ "$test_type" == "d2h" ]; then
        test_path=$D2H_PATH
    else
        test_path=$H2D_PATH
    fi

    echo "GPU${gpu_id} ${test_type^^} ${mode} (NUMA${numa_node} MEM${mem_node}) size=${data_size}"
    CUDA_VISIBLE_DEVICES=$gpu_id numactl -N $numa_node -m $mem_node \
        $test_path -s $data_size -n $iterations
}

# 多实例并发评估函数
eval_multi_instance() {
    local test_type=$1
    local data_size=$2
    local iterations=$3
    shift 3
    local instances=("$@")

    local test_path
    if [ "$test_type" == "d2h" ]; then
        test_path=$D2H_PATH
    else
        test_path=$H2D_PATH
    fi

    local tmp_dir=$(mktemp -d)
    local pid_list=""

    local idx=0
    for instance in "${instances[@]}"; do
        IFS=':' read -r gpu_id mode <<< "$instance"
        local numa_node=${GPU_NUMA_MAP[$gpu_id]}
        local mem_node
        if [ "$mode" == "local" ]; then
            mem_node=$numa_node
        elif [ "$mode" == "remote" ]; then
            mem_node=$((1 - numa_node))
        else
            echo "Error: mode must be 'local' or 'remote'"
            return 1
        fi
        local tmp_file="${tmp_dir}/gpu${gpu_id}_${idx}"
        
        CUDA_VISIBLE_DEVICES=$gpu_id numactl -N $numa_node -m $mem_node \
            $test_path -s $data_size -n $iterations > "$tmp_file" 2>&1 &
        
        local pid=$!
        pid_list="$pid_list $pid"
        ((idx++))
    done

    wait $pid_list

    echo "--- Results ---"
    for instance in "${instances[@]}"; do
        IFS=':' read -r gpu_id mode <<< "$instance"
        local tmp_file="${tmp_dir}/gpu${gpu_id}_0"
        if [ -f "$tmp_file" ]; then
            local result=$(grep "Throughput:" "$tmp_file")
            echo "GPU${gpu_id} ${mode}: $result"
        fi
    done

    rm -rf "$tmp_dir"
}

# 运行单实例测试
run_single_tests() {
    echo ""
    echo "============================================"
    echo "  单实例测试 (GPU ${SINGLE_GPU_ID})"
    echo "============================================"

    for test_type in "d2h" "h2d"; do
        for mode in "local" "remote"; do
            echo ""
            echo "--- ${test_type^^} ${mode} ---"
            for size in "${DATA_SIZES[@]}"; do
                eval_single_gpu $SINGLE_GPU_ID $mode $test_type $size $ITERATIONS
            done
        done
    done
}

# 运行多实例测试
run_multi_tests() {
    local instance_count=$1
    local -n instances_ref=$2
    local label=$3

    echo ""
    echo "============================================"
    echo "  ${instance_count}实例测试 (${label})"
    echo "============================================"

    for test_type in "d2h" "h2d"; do
        for instance_group in "${instances_ref[@]}"; do
            echo ""
            echo "--- ${test_type^^} ${instance_group} ---"
            for size in "${DATA_SIZES[@]}"; do
                eval_multi_instance $test_type $size $ITERATIONS $instance_group
            done
        done
    done
}

# 主函数
main() {
    echo "============================================"
    echo "  GPU D2H/H2D 性能评估"
    echo "  数据量: ${DATA_SIZES[*]}"
    echo "  迭代次数: ${ITERATIONS}"
    echo "============================================"

    # 单实例测试
    run_single_tests

    # 多实例测试
    run_multi_tests 2 MULTI_2GPU_INSTANCES "双实例"
    run_multi_tests 3 MULTI_3GPU_INSTANCES "三实例"
    run_multi_tests 4 MULTI_4GPU_INSTANCES "四实例"

    echo ""
    echo "============================================"
    echo "  评估完成！"
    echo "============================================"
}

main
