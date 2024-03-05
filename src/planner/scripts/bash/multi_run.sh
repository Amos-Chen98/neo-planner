#!/bin/bash
repeat_num=${1:-"2"} # Default to 1 if no argument is provided
echo "The number of repeats for each experiments is $repeat_num"

is_save_rosbag=${2:-"false"} # Default to true if no argument is provided
echo "The is_save_rosbag is $is_save_rosbag"

# selected_planner: 'basic', 'batch', 'record', 'nn', 'enhanced', or 'warmstart'
# replan_mode: 'global', 'online', or 'periodic'

min_test_unit() {
    local repeat_num=$1
    local world=$2

    # Loop to execute the script the specified number of times
    for ((i = 1; i <= repeat_num; i++)); do
        echo "========== Execution $i run  =========="

        echo "++++++++++ Execution started. ++++++++++"
        ./scripts/bash/demo_auto_stop.sh "nn" "$world" "periodic" "$is_save_rosbag"
        echo "++++++++++ Execution finished. ++++++++++"
        sleep 10

        echo "++++++++++ Execution started. ++++++++++"
        ./scripts/bash/demo_auto_stop.sh "basic" "$world" "periodic" "$is_save_rosbag"
        echo "++++++++++ Execution finished. ++++++++++"
        sleep 10

        echo "++++++++++ Execution started. ++++++++++"
        ./scripts/bash/demo_auto_stop.sh "batch" "$world" "periodic" "$is_save_rosbag"
        echo "++++++++++ Execution finished. ++++++++++"
        sleep 10

        echo "++++++++++ Execution started. ++++++++++"
        ./scripts/bash/demo_auto_stop.sh "enhanced" "$world" "periodic" "$is_save_rosbag"
        echo "++++++++++ Execution finished. ++++++++++"
        sleep 10
    done
}

# change worlds from here
min_test_unit $repeat_num "poles"
min_test_unit $repeat_num "forest"
min_test_unit $repeat_num "bricks"
min_test_unit $repeat_num "rand_world_5"
min_test_unit $repeat_num "rand_world_10"
min_test_unit $repeat_num "rand_world_15"
min_test_unit $repeat_num "rand_world_20"
min_test_unit $repeat_num "rand_world_25"
min_test_unit $repeat_num "rand_world_30"
min_test_unit $repeat_num "rand_world_35"
min_test_unit $repeat_num "rand_world_40"

# analysis
python3 scripts/bash/analyze_data.py
