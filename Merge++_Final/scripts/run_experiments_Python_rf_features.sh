#!/bin/bash
set -e

MODELS=("rf")
# METHODS=("chi2" "mutual_info" "rfe" "rf_importance")
METHODS=("baseline")
KS=(10 20 25 30 35 40)
LANGUAGE="Python"
EVAL_MODE="cross_project"

for MODEL in "${MODELS[@]}"; do
  mkdir -p "results/${MODEL}"
  mkdir -p "results/${LANGUAGE}"

  for METHOD in "${METHODS[@]}"; do

    if [[ "$METHOD" == "baseline" || "$METHOD" == "pca" ]]; then
      echo "Running $MODEL with $METHOD (no k)"
      python scripts/run_experiments.py \
        --model "$MODEL" \
        --method "$METHOD" \
        --language "$LANGUAGE" \
        --eval_mode "$EVAL_MODE" \
        > "results/${LANGUAGE}/${MODEL}_${METHOD}_${LANGUAGE}_${EVAL_MODE}.txt" 2>&1
      echo "  -> saved results/${LANGUAGE}/${MODEL}_${METHOD}_${LANGUAGE}_${EVAL_MODE}.txt"

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
          > "results/${LANGUAGE}/${MODEL}_${METHOD}_k${K}_${LANGUAGE}_${EVAL_MODE}.txt" 2>&1
        echo "  -> saved results/${LANGUAGE}/${MODEL}_${METHOD}_k${K}_${LANGUAGE}_${EVAL_MODE}.txt"
      done
    fi

  done
done

echo ""
echo "All experiments finished."