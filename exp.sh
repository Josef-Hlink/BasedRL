#!/bin/bash

# test different deltas

DEFAULTS="greek -PID greek -c 50"

for i in {0..10}; do
    pbrl-sweep $DEFAULTS -SID $i &
done

wait

echo "Done!"
