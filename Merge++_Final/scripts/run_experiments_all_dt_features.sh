#!/bin/bash
set -e

# MODELS=("rf" "lr" "dt" "xgb")
MODELS=("dt")
METHODS=("chi2" "mutual_info" "rfe" "rf_importance")
# METHODS=("baseline")
KS=(25 35)
LANGUAGE="all"
EVAL_MODE="global"

for MODEL in "${MODELS[@]}"; do
  mkdir -p "results/${MODEL}"

  for METHOD in "${METHODS[@]}"; do

    if [[ "$METHOD" == "baseline" || "$METHOD" == "pca" ]]; then
      echo "Running $MODEL with $METHOD (no k)"
      python scripts/run_experiments.py \
        --model "$MODEL" \
        --method "$METHOD" \
        --language "$LANGUAGE" \
        --eval_mode "$EVAL_MODE" \
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
          --language "$LANGUAGE" \
          --eval_mode "$EVAL_MODE" \
          > "results/${MODEL}/${MODEL}_${METHOD}_k${K}.txt" 2>&1
        echo "  -> saved results/${MODEL}/${MODEL}_${METHOD}_k${K}.txt"
      done
    fi

  done
done

echo ""
echo "All experiments finished."