#!/bin/bash
set -e

# MODELS=("rf" "lr" "dt" "xgb")
MODELS=("xgb")
METHODS=("baseline" "chi2" "mutual_info" "rfe" "rf_importance" "pca")
# METHODS=("pca")
KS=(10 20 30 40)

for MODEL in "${MODELS[@]}"; do
  mkdir -p "results/${MODEL}"

  for METHOD in "${METHODS[@]}"; do

    if [[ "$METHOD" == "baseline" || "$METHOD" == "pca" ]]; then
      echo "Running $MODEL with $METHOD (no k)"
      python scripts/run_experiments.py \
        --model "$MODEL" \
        --method "$METHOD" \
        > "results/${MODEL}/${MODEL}_${METHOD}.txt" 2>&1
      echo "  -> saved results/${MODEL}/${MODEL}_${METHOD}.txt"

    else
      for K in "${KS[@]}"; do
        echo "Running $MODEL with $METHOD k=$K"
        python scripts/run_experiments.py \
          --model "$MODEL" \
          --method "$METHOD" \
          --k "$K" \
          --save_freq \
          > "results/${MODEL}/${MODEL}_${METHOD}_k${K}.txt" 2>&1
        echo "  -> saved results/${MODEL}/${MODEL}_${METHOD}_k${K}.txt"
      done
    fi

  done
done

echo ""
echo "All experiments finished."