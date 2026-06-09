#!/bin/bash
# E4 BAD-branch autonomous pipeline (Lee2019 E4 already = BAD).
# Runs the pre-planned benchmark + salvage experiments, dependency-ordered, failure-isolated, logged.
# Self-contained: survives without the agent being re-invoked. User reviews the log when back.
cd /mnt/data/seiya.narukawa/intentflow || exit 1
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate intentflow
RES=intentflow/offline/results/research_outputs
A=intentflow/offline/scripts/analysis
H=$A/lee2019_e4_riskcoverage.py
LOG=$RES/260608_e4_pipeline.log
exec >>"$LOG" 2>&1
step(){ echo; echo "----- [$(date +%H:%M:%S)] $1 -----"; }
echo "==================== E4 BAD-branch AUTONOMOUS PIPELINE START $(date) ===================="
echo "Lee2019 E4 already ran = BAD (clusterability loses to nuc_dispersity). This pipeline = benchmark + FB4 salvage."

step "STEP1: build 2a per-trial npz (CPU, from existing verified features)"
python $A/build_2a_e4_npz.py || echo "!! STEP1 FAILED"

step "STEP2: 2a E4 risk-coverage (lda mode, k=4) -- does BAD generalize to 2a?"
python $H --npz $RES/260608_2a_pertrial.npz --out $RES/260608_2a_e4_riskcoverage.json || echo "!! STEP2 FAILED"

step "STEP3: dump 2b per-trial (GPU forward only, no training, ~minutes)"
CUDA_VISIBLE_DEVICES=1 python $A/dump_2b_e4_npz.py || echo "!! STEP3 FAILED"

step "STEP4: 2b E4 risk-coverage (lda mode, k=2) -- 2b is the HARD low-acc/3ch dataset"
python $H --npz $RES/260608_2b_pertrial.npz --out $RES/260608_2b_e4_riskcoverage.json || echo "!! STEP4 FAILED"

step "STEP5: FB4 adaptation-worthiness salvage (predict BENEFIT not accuracy; 2a DA-DC gain + 2b EA-benefit)"
python $A/fb4_adapt_worthiness.py || echo "!! STEP5 FAILED"

step "DONE"
echo "==================== PIPELINE DONE $(date) ===================="
echo "VERDICT SUMMARY (grep):"
grep -h ">>> VERDICT" $LOG | tail -n 6
echo "(Potato/R2 was intentionally deferred from the unattended run -- needs careful raw-EEG design when back.)"
