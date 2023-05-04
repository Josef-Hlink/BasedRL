#!/bin/bash

# test different deltas

DEFAULTS="reinforce -PID RF -c 50"

for i in {1..10}; do
    pbrl-sweep $DEFAULTS -SID $i &
    sleep 1
done

wait

echo "Done!"
