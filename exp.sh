#!/bin/bash

# project ID: EXP
# agent type: RF
# budget: 2500
# evaluation episodes: 250
# batch size: 64
# alpha: 0.0025
# beta: 0.1
# gamma: 0.8
# delta: 0.99
# quiet: true
# wandb: true

DEFAULTS="-PID EXP -at RF -bu 2500 -ee 250 -bs 64 -a 0.0025 -b 0.1 -g 0.8 -d 0.99 -Q -W"

# entropy regularization
# for b in 0.0 0.1 0.2 0.3; do
#     seq 1 10 | xargs -I{} -n 1 -P 10 pbrl-run $DEFAULTS -RID "b$b" -b $b
# done

DEFAULTS="$DEFAULTS -PID EXP3 -at AC"

# bootstrapping and baseline subtraction
for flags in "" "-bl" "-bo" "-bl -bo"; do
    seq 1 10 | xargs -I{} -n 1 -P 10 pbrl-run $DEFAULTS -RID "f$flags" $flags
done
