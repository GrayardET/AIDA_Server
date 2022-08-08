#!/bin/bash

TASK="nn"
python new_Switch.py nn0 1000 30000&
python read*no* nn1 0.07
for ((i=1;i <= 10 ;i++))
do
        NAME="$TASK$i"
        echo "$NAME"
        python new_Switch.py $NAME 1000 30000&
        sleep 3
done

#NAME="cpu3"
#python new_Switch.py $NAME 50000 60000&

