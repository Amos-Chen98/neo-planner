#!/bin/bash
num_repeats=${1:-"1"}  # Default to 1 if no argument is provided
echo "The number of repeats for each experiments is $num_repeats"

is_save_rosbag=${2:-"false"}  # Default to true if no argument is provided
echo "The is_save_rosbag is $is_save_rosbag"

# selected_planner: 'basic', 'batch', 'record', 'nn', 'enhanced', or 'warmstart'
# replan_mode: 'global', 'online', or 'periodic'

min_test_unit() {
  local num_runs=$1
  local world=$2

  # Loop to execute the script the specified number of times
  for (( i = 1; i <= num_runs; i++ ))
  do
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
    ./scripts/bash/demo_auto_stop.sh "enhanced" "$world" "periodic" "$is_save_rosbag"
    echo "++++++++++ Execution finished. ++++++++++"
    sleep 10
  done
}

# change worlds from here
min_test_unit $num_repeats "rand_world_10"
min_test_unit $num_repeats "rand_world_20"
min_test_unit $num_repeats "rand_world_40"
