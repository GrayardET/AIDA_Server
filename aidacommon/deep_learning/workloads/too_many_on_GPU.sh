#!/bin/bash

TASK="nn"
for ((i=1;i <= 10 ;i++))
do
        NAME="$TASK$i"
        echo "$NAME"
        python ../new_Switch.py $NAME 1000000 40000&
        sleep 3
done

NAME="cpu3"
python ../new_Switch.py $NAME 50000 60000&

