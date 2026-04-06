# Quick Start Guide - Clean Baseline Experiment

## 🚀 Start Experiment (Recommended Method)

### Option 1: Using the background script
```bash
cd /workspace-cloud/seiya.narukawa/intentflow/intentflow/offline
./scripts/run_experiment_background.sh
```

### Option 2: Manual start with tmux (Best for long experiments)
```bash
# Create a new tmux session
tmux new -s intentflow_experiment

# Inside tmux session:
cd /workspace-cloud/seiya.narukawa/intentflow/intentflow/offline
conda activate intentflow
python3 train_pipeline.py --model tcformer --dataset bcic2a --seed 0 --gpu_id 0

# Detach from tmux: Press Ctrl+B, then D
# Reattach later: tmux attach -t intentflow_experiment
```

### Option 3: Direct run (will occupy terminal)
```bash
cd /workspace-cloud/seiya.narukawa/intentflow/intentflow/offline
conda activate intentflow
python3 train_pipeline.py --model tcformer --dataset bcic2a --seed 0 --gpu_id 0
```

---

## 📊 Monitor Progress

### Check if experiment is running
```bash
ps aux | grep train_pipeline | grep -v grep
```

### Monitor log in real-time
```bash
# Find latest log file
ls -lht results/*.log | head -1

# Tail the log
tail -f results/run_clean_baseline_tcformer_s0_*.log
```

### Check GPU usage
```bash
watch -n 1 nvidia-smi
```

---

## 📈 Check Results

### After experiment completes
```bash
# Find latest results directory
ls -lhtd results/clean_baseline_tcformer_s0_* | head -1

# View results
cat results/clean_baseline_tcformer_s0_*/results.txt
```

### Expected output
```
Results for model: TCFormer
#Params: 77840
Dataset: bcic2a
Subject IDs: [1, 2, 3, 4, 5, 6, 7, 8, 9]

Results for each subject:
Subject 1 => Test Acc: 0.XXXX, Test Kappa: 0.XXXX
...

--- Summary Statistics ---
Average Test Accuracy: 83-85 ± 8-10  <- Expected improvement!
```

---

## 🔧 Troubleshooting

### Experiment stops immediately
**Cause**: Wrong Python version or conda env not activated

**Fix**:
```bash
conda activate intentflow
which python3  # Should show: ~/anaconda3/envs/intentflow/bin/python3
```

### CUDA errors
**Cause**: GPU busy or driver issue

**Fix**:
```bash
nvidia-smi  # Check GPU usage
# Try different GPU
python3 train_pipeline.py --model tcformer --dataset bcic2a --seed 0 --gpu_id 1
```

### Out of memory
**Fix**: Reduce batch size in config
```bash
# Edit tcformer.yaml
vim configs/tcformer/tcformer.yaml
# Change: batch_size: 48 -> batch_size: 32
```

---

## ⏱️ Expected Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| Setup | ~30 sec | Data loading, model initialization |
| Training | ~90-120 min | 1000 epochs × 9 subjects |
| Testing | ~2 min/subject | Final evaluation |
| **Total** | **~100-130 min** | Full experiment |

---

## ✅ Success Indicators

During training, you should see:
- ✅ "Split session_T: train=230, val=58, test=288"
- ✅ "Epoch X: train_loss=..., val_loss=..., train_acc=..., val_acc=..."
- ✅ NO "Signaling Trainer to stop" message
- ✅ Training reaches Epoch 999

After completion:
- ✅ results.txt file created
- ✅ Average accuracy: **83-85%** (improved from 80.98%)
- ✅ Standard deviation: **8-10%** (improved from 12.05%)

---

## 🎯 Next Steps

### If accuracy ≥ 83%
```bash
# Run 5-seed experiment
./scripts/run_clean_baseline_5seeds.sh
```

### If accuracy < 83%
Investigate:
1. Window timing (0.0-4.0s vs 0.5-4.5s)
2. Transformer depth (2 vs 4)
3. Learning rate schedule

---

## 📁 Generated Files

```
results/clean_baseline_tcformer_s0_TIMESTAMP/
├── config.yaml              # Experiment configuration
├── results.txt              # Summary results
├── final_acc_TCFormer.json  # Accuracy per subject
├── checkpoints/             # Saved model weights
│   ├── subject_1_model.ckpt
│   └── ...
├── confmats/                # Confusion matrices
│   ├── confmat_subject_1.png
│   ├── avg_confusion_matrix.png
│   └── ...
└── curves/                  # Training curves
    ├── subject_1_loss.png
    ├── subject_1_acc.png
    └── ...
```

---

## 💡 Tips

1. **Use tmux** for long experiments - prevents disconnection issues
2. **Monitor GPU** regularly - ensures training is progressing
3. **Check logs** periodically - catch errors early
4. **Save results** - copy important files before next experiment

---

## 📞 Help

If you encounter issues:
1. Check [FIXES_20260306.md](FIXES_20260306.md) for known issues
2. Check [EXPERIMENT_STATUS.md](EXPERIMENT_STATUS.md) for current status
3. Review logs for error messages
