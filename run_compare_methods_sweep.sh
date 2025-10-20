#!/bin/bash
# ==========================
# run_sweep.sh
# Balaye plusieurs combinaisons d'hyperparamètres
# et trace les résultats après chaque run.
# ==========================

# --- Listes de paramètres à tester ---
LAMBDAS=(0.1 0.01 0.001)
STEPSIZES=(0.1 0.05)
NCOARSE_STEPS=(5 10)
J_LEVELS=(3 4)
IMAGE_SIZE="small"
METHODS="FB BCD_FB MLFBcond"
PHYSICS="deblurring"
PRIOR="L1_wavelet"
SIGMA=0.01
N_ITER=100

# --- Boucles de balayage ---
for lambda in "${LAMBDAS[@]}"; do
  for step in "${STEPSIZES[@]}"; do
    for ncoarse in "${NCOARSE_STEPS[@]}"; do
      for J in "${J_LEVELS[@]}"; do
        echo "=============================================="
        echo ">>> λ=$lambda | step=$step | n_coarse=$ncoarse | J=$J"
        echo "=============================================="

        python3 compare_methods.py \
          --physics "$PHYSICS" \
          --prior "$PRIOR" \
          --reg_weight "$lambda" \
          --stepsize "$step" \
          --sigma "$SIGMA" \
          --J "$J" \
          --n_iter "$N_ITER" \
          --n_coarse_steps "$ncoarse" \
          --image_size "$IMAGE_SIZE" \
          --methods $METHODS \
          > "logs/run_lambda${lambda}_step${step}_coarse${ncoarse}_J${J}.log" 2>&1

        # Récupère le dernier dossier d'expérience
        EXP_DIR=$(ls -td /projects/users/edesaint/IMLFISTA/experiments_results/compare_methods/exp_* | head -1)

        echo "Plotting results for: $EXP_DIR"
        python3 plot_results.py "$EXP_DIR"

        echo ""
        echo "Done λ=$lambda, step=$step, n_coarse=$ncoarse, J=$J"
        echo ""
      done
    done
  done
done

echo "Sweep done."
