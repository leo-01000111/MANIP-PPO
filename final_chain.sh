#!/bin/bash
# Final runs for the report: position stage from scratch, then the pose stage (warm start).
# Usage: bash final_chain.sh NAME SEED "extra flags for the agent"   (run from the repo root)
# Example: bash final_chain.sh F_q 0 ""     bash final_chain.sh F_noq_mem 0 "--no-obs-q --history 4"
NAME=$1; SEED=$2; EXTRA=$3
COMMON="--seed $SEED --n-envs 4 --threads 2 --eval-freq 200000 --no-sde --gamma 0.995 --max-steps 200 --time-penalty 0.01"
POS="${NAME}_s${SEED}_pos"; POSE="${NAME}_s${SEED}_pose"

python src/train.py --name $POS $COMMON $EXTRA --steps 18000000 \
    --n-steps 1024 --batch-size 512 --target-kl 0.02 --lr-decay > runs/$POS.log 2>&1

if [ -f runs/$POS/final_model.zip ]; then
    python src/train.py --name $POSE $COMMON $EXTRA --steps 12000000 --init-from runs/$POS/final_model.zip \
        --task pose --w-rot 1.0 --reset-std 0.37 --rot-tol-start 45 --tol-frac 0.5 --target-rot-range 60 \
        > runs/$POSE.log 2>&1
else
    echo "position stage of $POS failed, pose stage not started" >> runs/$POS.log
fi
