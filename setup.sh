#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  setup.sh  —  Initialise the DRexInference repo after cloning
#
#  Run once:
#    bash setup.sh
#
#  What it does:
#    1. Initialises and checks out the cosmos_transfer1 submodule
#    2. Applies local patches to cosmos_transfer1
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "── Step 1: initialise submodule ─────────────────────────────────────────"
git submodule update --init --recursive

echo ""
echo "── Step 2: apply cosmos_transfer1 patches ───────────────────────────────"
PATCH="$REPO_ROOT/patches/cosmos_transfer1.patch"
if [ ! -f "$PATCH" ]; then
    echo "[ERROR] Patch file not found: $PATCH"
    exit 1
fi

cd "$REPO_ROOT/cosmos_transfer1"
if git apply --check "$PATCH" 2>/dev/null; then
    git apply "$PATCH"
    echo "[OK] Patch applied."
else
    echo "[WARN] Patch does not apply cleanly — may already be applied or conflicts exist."
    echo "       Run 'git apply --reject $PATCH' inside cosmos_transfer1/ to inspect."
fi

cd "$REPO_ROOT"
echo ""
echo "── Setup complete ────────────────────────────────────────────────────────"
echo "   Activate your environment and run:"
echo "     conda activate cosmos-predict1"
echo "     CUDA_HOME=\$CONDA_PREFIX PYTHONPATH=.:cosmos_transfer1 python scripts/relight.py --help"
