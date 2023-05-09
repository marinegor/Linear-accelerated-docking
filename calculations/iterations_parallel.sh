#!/bin/bash

methods=("MeanRank" "LastModel" "TopFromEveryModel") # list methods that you want to run
FOUT='/storage/margaritakovaleva/2022-07-22_tables/4eiy/tables_for_prediction/Morgan_size=2048_radius=2'

for method in MeanRank LastModel TopFromEveryModel
    for iterations in 30 24 12 6; do
        python Iterations.py \
            -path "${FOUT}" \
            --method ${method} \
            --model LinearSVR \
            --folds 3 -i ${iterations} -ts 240000
done &
done
