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
#SBATCH --mem=32G
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
# ---------------------------------
# Copy script input data and go to working directory
# ATTENTION : Il faut que le script soit dans le répertoire de travail

# Chargement du module go
# --- Go / build toolchain: prefer spack module, sinon fallback local ---
GO_MODULE="go@1.23.1"
PYTHON_MODULE="python@3.11.9"
SETUP_TOOLS_MODULE="py-setuptools@67.6.0"
PIP_MODULE="py-pip@23.0"
CUDA_MODULE="cuda@12.6.2"

# try loading spack modules
if spack load "${CUDA_MODULE}" 2>/dev/null; then
  echo "[INFO] Loaded ${CUDA_MODULE} via spack."
else
  echo "[WARN] spack couldn't load ${CUDA_MODULE}."
fi
if spack load "${GO_MODULE}" 2>/dev/null; then
  echo "[INFO] Loaded ${GO_MODULE} via spack."
else
  echo "[WARN] spack couldn't load ${GO_MODULE}."
fi
if spack load "${PYTHON_MODULE}" 2>/dev/null; then
  echo "[INFO] Loaded ${PYTHON_MODULE} via spack."
else
  echo "[WARN] spack couldn't load ${PYTHON_MODULE}."
fi
if spack load "${SETUP_TOOLS_MODULE}" 2>/dev/null; then
  echo "[INFO] Loaded ${SETUP_TOOLS_MODULE} via spack."
else
  echo "[WARN] spack couldn't load ${SETUP_TOOLS_MODULE}."
fi
if spack load "${PIP_MODULE}" 2>/dev/null; then
  echo "[INFO] Loaded ${PIP_MODULE} via spack."
else
  echo "[WARN] spack couldn't load ${PIP_MODULE}."
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

echo "[INFO] Using python at $PYTHON_BIN"

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
# Désinstallez la librairie publique si elle est présente
pip uninstall orion -y

# Désinstallez votre version actuelle pour repartir au propre
pip uninstall orion-fhe -y
pip install -e .
echo "orion -e . installed."

echo "Job started at `date`"

# assure-toi que VENV_DIR est défini plus haut (ex: VENV_DIR="${SLURM_TMPDIR:-$HOME/.cache}/bcresnet_venv")
srun bash -lc "source '$VENV_DIR/bin/activate' && echo 'Using python: ' \$(which python) && python -c 'import orion; print(\"orion:\", getattr(orion,\"__file__\",None))' && cd examples && python -u run_resnet.py"

echo "Job finished at `date`"

exit 0
# End of job script
# ---------------------------------