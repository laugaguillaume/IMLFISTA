#!/bin/bash
# ==========================
# Run one experiment comparing multiple methods, plot results
# ==========================

# --- Paramètres ---
PHYSICS="deblurring"          # or "inpainting"
PRIOR="L1_wavelet"            # or "L1", "TV"
REG_WEIGHT=0.001               # lambda
STEPSIZE=0.1
SIGMA=0.01
J=1
N_ITER=100
N_COARSE_STEPS=5
IMAGE_SIZE="small"            # or "big"
METHODS="FB"         # space separated (FB MLFB PnP MLPnP MLFBcond BCD_FB BCD_MLFB BCDcyclic BCD_MLFB_details BCDcyclic_cond)
# ==========================

# --- Main call ---
echo "Running experiment with:"
echo "  physics=$PHYSICS, prior=$PRIOR, reg_weight=$REG_WEIGHT, stepsize=$STEPSIZE"
echo "  sigma=$SIGMA, J=$J, n_iter=$N_ITER, n_coarse_steps=$N_COARSE_STEPS, image_size=$IMAGE_SIZE"
echo "  methods=$METHODS"
echo ""

python3 compare_methods.py \
  --physics "$PHYSICS" \
  --prior "$PRIOR" \
  --reg_weight "$REG_WEIGHT" \
  --stepsize "$STEPSIZE" \
  --sigma "$SIGMA" \
  --J "$J" \
  --n_iter "$N_ITER" \
  --n_coarse_steps "$N_COARSE_STEPS" \
  --image_size "$IMAGE_SIZE" \
  --methods $METHODS

# --- Identify experiment directory ---
EXP_DIR=$(ls -td experiments_results/compare_methods/exp_* | head -1)

echo ""
echo "Done. Results stored in:"
echo "   $EXP_DIR"
echo "Plotting results..."

python3 plot_results.py "$EXP_DIR"
