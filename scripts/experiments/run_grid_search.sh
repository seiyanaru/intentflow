#!/bin/bash
set -e
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "=== Hybrid Model Grid Search ==="

# Define parameter grid
LR_LIST=(0.1 0.01)
REG_LIST=(0.01 0.05 0.1)
RATIO_LIST=(0.25 0.125)

# Output CSV for results summary
SUMMARY_FILE="intentflow/offline/analysis/grid_search_summary.csv"
echo "LR,Reg,Ratio,Subject,Accuracy" > $SUMMARY_FILE

for lr in "${LR_LIST[@]}"; do
  for reg in "${REG_LIST[@]}"; do
    for ratio in "${RATIO_LIST[@]}"; do
      
      EXP_NAME="Hybrid_LR${lr}_Reg${reg}_Ratio${ratio}"
      echo "----------------------------------------------------------------"
      echo "Running Experiment: $EXP_NAME"
      echo "  Base LR: $lr"
      echo "  Reg Lambda: $reg"
      echo "  Adapter Ratio: $ratio"
      
      # Construct model_kwargs override JSON
      # Note: We use single quotes for shell string, and escape inner quotes for JSON
      KWARGS="{\"ttt_config\": {\"base_lr\": $lr, \"ttt_reg_lambda\": $reg}, \"adapter_ratio\": $ratio}"
      
      # Run Training (All subjects)
      # Using --seed 42 for consistency
      python intentflow/offline/train_pipeline.py \
          --model tcformer_hybrid \
          --dataset bcic2a \
          --gpu_id 0 \
          --seed 42 \
          --model_kwargs "$KWARGS" 2>&1 | tee "intentflow/offline/analysis/log_${EXP_NAME}.txt"
      
      # Find the result directory for this run (most recent one)
      RESULT_DIR=$(ls -td intentflow/offline/results/TCFormer_Hybrid_bcic2a_seed-42_* | head -1)
      
      echo "  Saved to: $RESULT_DIR"
      
      # Extract Accuracies from results.txt and append to summary
      RES_TXT="${RESULT_DIR}/results.txt"
      if [ -f "$RES_TXT" ]; then
          # Extract subject accuracies using grep/awk
          # Format in results.txt: "Subject  1 => ... Test Acc: 0.8500"
          grep "Subject" "$RES_TXT" | grep "Test Acc" | while read -r line ; do
              subj=$(echo "$line" | grep -oP "Subject\s+\d+" | grep -oP "\d+")
              acc=$(echo "$line" | grep -oP "Test Acc:\s+\d+\.\d+" | grep -oP "\d+\.\d+")
              echo "$lr,$reg,$ratio,$subj,$acc" >> $SUMMARY_FILE
          done
          
          # Print average for quick check
          AVG_ACC=$(grep "Average Test Accuracy" "$RES_TXT" | grep -oP "\d+\.\d+")
          echo "  -> Average Accuracy: $AVG_ACC"
      else
          echo "  Error: results.txt not found."
      fi
      
    done
  done
done

echo "Grid search complete. Summary saved to $SUMMARY_FILE"

