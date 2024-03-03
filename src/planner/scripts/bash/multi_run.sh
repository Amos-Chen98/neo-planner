#!/bin/bash

echo "++++++++++ run ./scripts/bash/demo_auto_stop.sh poles ++++++++++"
./scripts/bash/demo_auto_stop.sh poles
echo "++++++++++ Execution finished. ++++++++++"

sleep 10

echo "++++++++++ run ./scripts/bash/demo_auto_stop.sh bricks ++++++++++"
./scripts/bash/demo_auto_stop.sh bricks
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
