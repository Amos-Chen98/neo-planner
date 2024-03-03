#!/bin/bash
is_save_rosbag=${2:-"false"}  # Default to true if no argument is provided
echo "The is_save_rosbag is $is_save_rosbag"

echo "++++++++++ Execution stated. ++++++++++"
./scripts/bash/demo_auto_stop.sh "rand_world_10" "$is_save_rosbag"
echo "++++++++++ Execution finished. ++++++++++"

sleep 10

echo "++++++++++ Execution stated. ++++++++++"
./scripts/bash/demo_auto_stop.sh "rand_world_20" "$is_save_rosbag"
echo "++++++++++ Execution finished. ++++++++++"

sleep 10

echo "++++++++++ Execution stated. ++++++++++"
./scripts/bash/demo_auto_stop.sh "rand_world_40" "$is_save_rosbag"
echo "++++++++++ Execution finished. ++++++++++"

## Check if the number of repetitions is provided as an argument
#if [ -z "$1" ]; then
#  echo "Please provide the number of times you want to repeat the script."
#  echo "Usage: $0 <number-of-times>"
#  exit 1
#fi
#
## Extract the number of repetitions
#num_repeats=$1
#
## Loop to execute the script the specified number of times
#for (( i = 1; i <= num_repeats; i++ ))
#do
#  echo "Execution $i of ./scripts/bash/demo_auto_stop.sh bricks"
#  ./scripts/bash/demo_auto_stop.sh bricks
#  echo "Execution $i finished."
#done
