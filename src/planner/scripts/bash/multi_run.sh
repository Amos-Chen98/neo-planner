#!/bin/bash
repeat_num=${1:-"2"} # Default to 1 if no argument is provided
echo "The number of repeats for each experiments is $repeat_num"

is_save_rosbag=${2:-"false"} # Default to true if no argument is provided
echo "The is_save_rosbag is $is_save_rosbag"

# selected_planner: 'basic', 'batch', 'record', 'nn', 'neo', or 'warmstart'
# replan_mode: 'global', 'online', or 'periodic'

min_test_unit() {
    local repeat_num=$1
    local world=$2
    local num_models=$3

    # Loop to execute the script the specified number of times
    for ((i = 1; i <= repeat_num; i++)); do
        echo "========== Execution $i run  =========="

        echo "++++++++++ Execution started. ++++++++++"
        ./scripts/bash/demo_auto_stop.sh "nn" "periodic" "$is_save_rosbag" "$world" "$num_models"
        echo "++++++++++ Execution finished. ++++++++++"
        sleep 15

        echo "++++++++++ Execution started. ++++++++++"
        ./scripts/bash/demo_auto_stop.sh "basic" "periodic" "$is_save_rosbag" "$world" "$num_models"
        echo "++++++++++ Execution finished. ++++++++++"
        sleep 15

        echo "++++++++++ Execution started. ++++++++++"
        ./scripts/bash/demo_auto_stop.sh "batch" "periodic" "$is_save_rosbag" "$world" "$num_models"
        echo "++++++++++ Execution finished. ++++++++++"
        sleep 15

        echo "++++++++++ Execution started. ++++++++++"
        ./scripts/bash/demo_auto_stop.sh "neo" "periodic" "$is_save_rosbag" "$world" "$num_models"
        echo "++++++++++ Execution finished. ++++++++++"
        sleep 15
    done
}

# change worlds from here
min_test_unit "$repeat_num" "rand_world_10(1)" 10
min_test_unit "$repeat_num" "rand_world_10(2)" 10
min_test_unit "$repeat_num" "rand_world_15(1)" 15
min_test_unit "$repeat_num" "rand_world_20(1)" 20

# analysis
python3 scripts/bash/analyze_data.py
