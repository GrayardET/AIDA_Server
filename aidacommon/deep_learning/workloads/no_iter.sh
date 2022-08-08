#!/bin/bash

TASK="nn"
#python ../read_from*no* long1 0.07&
python ../new_Switch_no_iter.py i1 1000 10&

sleep 4
for ((i=1;i <= 3;i++))
do
        NAME="$TASK$i"
        echo "$NAME"
        python ../new_Switch.py $NAME 1000 30000&
        sleep 4
done

