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

# Installation de go (à faire une seule fois par machine)
cd /tmp
wget https://go.dev/dl/go1.22.3.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.22.3.linux-amd64.tar.gz
echo 'export PATH=/usr/local/go/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
go version # go version go1.22.3 linux/amd64

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
  pip install orion
  pip install -e .
  echo "orion installed."
fi

echo "[INFO] Using python at $PYTHON_BIN"

ls
cd examples/
srun $PYTHON_BIN -u run_resnet.py

echo "Job finished at `date`"

exit 0
# End of job script
# ---------------------------------