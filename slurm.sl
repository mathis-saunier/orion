#!/bin/bash

# Slurm submission script, 
# torch job 
# CRIHAN v 1.00 - Jan 2023 
# support@criann.fr


# Job name
#SBATCH -J "main"

# Batch output file
#SBATCH --output main.o%J

# Batch error file
#SBATCH --error main.e%J

# GPUs architecture and number
# ----------------------------
# Partition (submission class)
#SBATCH --partition mesonet

# ----------------------------
# processes / tasks
#SBATCH -n 1

# ------------------------
# Job time (hh:mm:ss)
#SBATCH --time 1:00:00
# ------------------------

#SBATCH --account=m25206

##SBATCH --mail-type ALL
# User e-mail address
#SBATCH --mail-user mathis.saunier@insa-rouen.fr


## Job script ##
# environments
# ---------------------------------
# pip install --user --no-cache-dir --quiet \
#    PyYAML \
#    torch \
#    torchvision \
#    tqdm \
#    numpy \
#    scipy \
#    matplotlib \
#    h5py \
#    certifi

# ---------------------------------
# Copy script input data and go to working directory
# ATTENTION : Il faut que le script soit dans le répertoire de travail

# Chargement du module go
# --- Go / build toolchain: prefer spack module, sinon fallback local ---
GO_MODULE="go@1.23.1"

# try loading spack go; if it fails, try go@1.21 or fallback to local install
if spack load "${GO_MODULE}" 2>/dev/null; then
  echo "[INFO] Loaded ${GO_MODULE} via spack."
else
  echo "[WARN] spack couldn't load ${GO_MODULE}. Trying go@1.21..."
  if ! spack load "go@1.21" 2>/dev/null; then
    echo "[WARN] No suitable spack go found — installing go ${GO_MODULE#go@} locally in \$HOME/.local/go"
    GO_VERSION="${GO_MODULE#go@}"   # e.g. 1.23.1
    ARCH="$(uname -m)"
    # choose tarball name (x86_64 -> amd64)
    case "$ARCH" in
      x86_64|amd64) TARCH="amd64" ;;
      aarch64|arm64) TARCH="arm64" ;;
      *) echo "[ERROR] arch $ARCH not handled"; exit 1 ;;
    esac
    TAR="go${GO_VERSION}.linux-${TARCH}.tar.gz"
    cd /tmp || exit 1
    curl -sSLO "https://dl.google.com/go/${TAR}" || { echo "[ERROR] curl failed"; exit 1; }
    mkdir -p "$HOME/.local"
    tar -C "$HOME/.local" -xzf "${TAR}"
    export GOROOT="$HOME/.local/go"
    export PATH="$GOROOT/bin:$PATH"
    echo "[INFO] Installed local go at $GOROOT"
  else
    echo "[INFO] Loaded go@1.21 via spack (may be OK if repo needs >=1.21)."
  fi
fi

# Ensure gcc is available for cgo (adjust module name if your cluster uses 'module' instead of spack)
if ! command -v gcc >/dev/null 2>&1; then
  if spack load gcc >/dev/null 2>&1; then
    echo "[INFO] Loaded gcc via spack."
  else
    module load gcc 2>/dev/null || echo "[WARN] gcc module not found — ensure gcc is available."
  fi
fi

# Environment fixes for building Go c-shared libs
export CGO_ENABLED=1
export CC="$(command -v gcc || true)"
# Force use of local toolchain (prevents some automatic toolchain downloads)
export GOTOOLCHAIN=local

# Debug prints
echo "[INFO] go -> $(command -v go || echo 'no-go')"
go version || true
echo "[INFO] gcc -> $CC"
gcc --version | head -n 1 || true


rsync -av --exclude 'saved' ./ $LOCAL_WORK_DIR
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1

echo Working directory : $PWD
echo "Job started at `date`"

# Last-resort: build a venv with system python if still empty
if [ -z "$PYTHON_BIN" ]; then
  if command -v module >/dev/null 2>&1; then
    module load python/3.11.9 2>/dev/null || module load python/3.10.10 2>/dev/null || true
  fi
  BASE_PY="$(command -v python3 || true)"
  if [ -z "$BASE_PY" ]; then
    echo "[ERROR] Aucun python3 trouvÃ©. Fixez PYTHON_BIN (ex: ~/miniconda3/envs/h2ogpt/bin/python)." >&2
    echo "[DEBUG] Tried candidates: ${PYTHON_CANDIDATES[*]}" >&2
    exit 1
  fi
  VENV_DIR="${SLURM_TMPDIR:-$HOME/.cache}/bcresnet_venv"
  "${BASE_PY}" -m venv "$VENV_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  pip install --upgrade pip >/dev/null
  pip install PyYAML
  echo "PyYAML installed."
  pip install torch
  echo "torch installed."
  pip install torchvision
  echo "torchvision installed."
  pip install tqdm numpy
  echo "tqdm and numpy installed."
  pip install scipy matplotlib h5py certifi >/dev/null
  echo "scipy, matplotlib, h5py, and certifi installed."
  PYTHON_BIN="$VENV_DIR/bin/python"

  # après activation du venv (déjà dans ton script)
  echo "[DEBUG] go binary used: $(which go || echo no-go)"
  go version || echo "go not available — aborting pip build (or will fail)."

  # build install
  pip install -e . -v
  echo "orion -e . installed."

fi

echo "[INFO] Using python at $PYTHON_BIN"

# assure-toi que VENV_DIR est défini plus haut (ex: VENV_DIR="${SLURM_TMPDIR:-$HOME/.cache}/bcresnet_venv")
srun bash -lc "source '$VENV_DIR/bin/activate' && echo 'Using python: ' \$(which python) && python -c 'import orion; print(\"orion:\", getattr(orion,\"__file__\",None))' && cd examples && python -u run_resnet.py"


echo "Job finished at `date`"

exit 0
# End of job script
# ---------------------------------