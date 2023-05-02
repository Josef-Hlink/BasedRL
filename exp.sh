#!/bin/bash

# test different deltas

DEFAULTS="-V -ne 1000 -a 0.0005 -g 0.8 -PID delta"

for d in 0.995 0.996 0.997 0.998 0.999; do
    pbrl $DEFAULTS -d $d -RID d$d;
done
