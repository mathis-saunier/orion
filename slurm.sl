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

##SBATCH --mail-type ALL
# User e-mail address
#SBATCH --mail-user mathis.saunier@insa-rouen.fr


## Job script ##
# environments
# ---------------------------------
pip install --user --no-cache-dir --quiet \
    PyYAML \
    torch \
    torchvision \
    tqdm \
    numpy \
    scipy \
    matplotlib \
    h5py \
    certifi

# ---------------------------------
# Copy script input data and go to working directory
# ATTENTION : Il faut que le script soit dans le répertoire de travail
rsync -av --exclude 'saved' ./ $LOCAL_WORK_DIR
cd $LOCAL_WORK_DIR/

echo Working directory : $PWD
echo "Job started at `date`"

module list

echo "Librairies PIP"
pip list

echo "Lancement du script"

cd examples/
srun python -u run_resnet.py

echo "Job finished at `date`"

exit 0
# End of job script
# ---------------------------------