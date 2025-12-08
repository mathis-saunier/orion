#!/bin/bash
# Slurm submission script - orion run_resnet wrapper
# Mathis Saunier - adapted

#SBATCH -J "main"
#SBATCH --output main.o%J
#SBATCH --error main.e%J
#SBATCH --partition mesonet
#SBATCH -n 1
#SBATCH --time 1:00:00
#SBATCH --account=m25206
#SBATCH --mail-user mathis.saunier@insa-rouen.fr

set -euo pipefail
IFS=$'\n\t'

echo "Job started at $(date)"
echo "SLURM_SUBMIT_DIR = ${SLURM_SUBMIT_DIR:-$PWD}"
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# ----------------------------
# Default locations (modifiable à l'export avant sbatch)
# ----------------------------
# emplacement par défaut du venv : SLURM_TMPDIR si/job-local, sinon $HOME/.cache
VENV_DIR="${VENV_DIR:-${SLURM_TMPDIR:-$HOME/.cache}/bcresnet_venv}"
# Optionnel: chemin local de travail (rsync)
LOCAL_WORK_DIR="${LOCAL_WORK_DIR:-$PWD}"
# optionnel: si tu veux forcer un PYTHON_BIN avant sbatch, export PYTHON_BIN
PYTHON_BIN="${PYTHON_BIN:-}"

echo "[INFO] VENV_DIR = $VENV_DIR"
echo "[INFO] LOCAL_WORK_DIR = $LOCAL_WORK_DIR"

# ----------------------------
# Go toolchain: prefer spack module, sinon fallback local install
# ----------------------------
GO_MODULE="${GO_MODULE:-go@1.23.1}"

echo "[INFO] Trying to load Go (preferred: $GO_MODULE)"
if command -v spack >/dev/null 2>&1 && spack load "${GO_MODULE}" 2>/dev/null; then
  echo "[INFO] Loaded ${GO_MODULE} via spack."
else
  echo "[WARN] spack couldn't load ${GO_MODULE} (or spack missing). Trying go@1.21 via spack..."
  if command -v spack >/dev/null 2>&1 && spack load "go@1.21" 2>/dev/null; then
    echo "[INFO] Loaded go@1.21 via spack."
  else
    # fallback local install to $HOME/.local/go (no sudo)
    echo "[WARN] Installing Go locally to \$HOME/.local/go"
    GO_VERSION="${GO_MODULE#go@}"   # ex: 1.23.1
    ARCH="$(uname -m)"
    case "$ARCH" in
      x86_64|amd64) TARCH="amd64" ;;
      aarch64|arm64) TARCH="arm64" ;;
      *) echo "[ERROR] arch $ARCH not handled"; exit 1 ;;
    esac
    TAR="go${GO_VERSION}.linux-${TARCH}.tar.gz"
    cd /tmp
    if ! curl -sSLO "https://dl.google.com/go/${TAR}"; then
      echo "[ERROR] Download failed for ${TAR} — check go version or network"; exit 1
    fi
    mkdir -p "$HOME/.local"
    tar -C "$HOME/.local" -xzf "${TAR}"
    export GOROOT="$HOME/.local/go"
    export PATH="$GOROOT/bin:$PATH"
    cd "${SLURM_SUBMIT_DIR:-$PWD}"
    echo "[INFO] Installed local go at $GOROOT"
  fi
fi

# Ensure gcc available for cgo
if ! command -v gcc >/dev/null 2>&1; then
  if command -v spack >/dev/null 2>&1 && spack load gcc >/dev/null 2>&1; then
    echo "[INFO] Loaded gcc via spack."
  else
    # try module load (clusters vary)
    module load gcc 2>/dev/null || echo "[WARN] gcc not found via spack/module — ensure gcc exists for cgo"
  fi
fi

export CGO_ENABLED=1
export CC="$(command -v gcc || true)"
export GOTOOLCHAIN="${GOTOOLCHAIN:-local}"

echo "[DEBUG] go -> $(command -v go || echo 'no-go')"
go version || true
echo "[DEBUG] gcc -> $CC"
gcc --version | head -n 1 || true

# ----------------------------
# Sync workdir if requested (keeps user's original behavior)
# ----------------------------
rsync -av --exclude 'saved' ./ "$LOCAL_WORK_DIR" || true
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# ----------------------------
# Python / venv setup
# ----------------------------
# If PYTHON_BIN is already set by the caller, try infering the venv path
if [ -n "${PYTHON_BIN:-}" ] && [ -z "${_VENV_INFERRED:-}" ]; then
  BIN_DIR="$(dirname "$PYTHON_BIN")" || BIN_DIR=""
  POSS_VENV="$(dirname "$BIN_DIR")"
  if [ -x "$POSS_VENV/bin/activate" ]; then
    VENV_DIR="$POSS_VENV"
    _VENV_INFERRED=1
    echo "[INFO] Inferred VENV_DIR from PYTHON_BIN -> $VENV_DIR"
  fi
  unset BIN_DIR POSS_VENV
fi

# If no PYTHON_BIN provided, create a venv in VENV_DIR and activate it
if [ -z "${PYTHON_BIN:-}" ]; then
  echo "[INFO] No PYTHON_BIN provided — creating venv in $VENV_DIR"
  # try load system python modules if available
  if command -v module >/dev/null 2>&1; then
    module load python/3.11.9 2>/dev/null || module load python/3.10.10 2>/dev/null || true
  fi
  BASE_PY="$(command -v python3 || true)"
  if [ -z "$BASE_PY" ]; then
    echo "[ERROR] Aucun python3 trouvé. Exportez PYTHON_BIN ou installez python sur le noeud." >&2
    exit 1
  fi
  mkdir -p "$(dirname "$VENV_DIR")"
  "$BASE_PY" -m venv "$VENV_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  pip install --upgrade pip setuptools wheel >/dev/null
  # lightweight installs (non exhaustive)
  pip install PyYAML tqdm numpy >/dev/null
  pip install torch torchvision >/dev/null || echo "[WARN] torch install failed or slow; ensure your cluster has access to appropriate wheels"
  pip install scipy matplotlib h5py certifi >/dev/null || true
  PYTHON_BIN="$VENV_DIR/bin/python"
  echo "[INFO] Created and activated venv; PYTHON_BIN = $PYTHON_BIN"

  # confirm go visible for build
  echo "[DEBUG] go binary used: $(which go || echo no-go)"
  go version || echo "go not available — go builds may fail."

  # build and install orion editable
  echo "[INFO] Installing package in editable mode (pip install -e . -v)"
  pip install -e . -v
  echo "[INFO] pip install -e . finished"
else
  echo "[INFO] Using provided PYTHON_BIN = $PYTHON_BIN"
fi

echo "[INFO] Final PYTHON_BIN = $PYTHON_BIN"
"$PYTHON_BIN" -c "import sys; print('Python executable:', sys.executable); import pkgutil; print('orion present (iter_modules):', any(m.name=='orion' for m in pkgutil.iter_modules()))" || true

# ----------------------------
# Run examples/run_resnet.py on compute node (robuste)
# ----------------------------
# srun will attempt to activate the venv on the compute node; if impossible, fallback to direct binary
if srun bash -lc "test -e '$VENV_DIR/bin/activate' >/dev/null 2>&1"; then
  echo "[INFO] VENV exists on compute node — activating and running"
  srun bash -lc "source '$VENV_DIR/bin/activate' && echo 'Using python: ' \$(which python) && python -c 'import orion; print(\"orion:\", getattr(orion,\"__file__\",None))' && cd examples && python -u run_resnet.py"
else
  echo "[WARN] VENV not present on compute node; falling back to using $PYTHON_BIN directly"
  # ensure path is absolute and accessible on the node
  if [ -x "${PYTHON_BIN:-/usr/bin/python3}" ]; then
    srun "${PYTHON_BIN}" -u examples/run_resnet.py
  else
    echo "[ERROR] $PYTHON_BIN not executable on compute node; cannot run job." >&2
    exit 1
  fi
fi

echo "Job finished at $(date)"
exit 0
