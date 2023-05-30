#!/bin/bash

#SBATCH --job-name=inference_colo         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=128              # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=2G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:30:00          # total run time limit (HH:MM:SS)

module use --append /scratch/gpfs/aeshao/local/modulefiles
module load intel/2021.1 intel-mpi/intel/2021.1.1
module load cudatoolkit/11.7 cudnn/8.9.1
module load anaconda3/2023.3
module load netcdf/intel-2021.1/hdf5-1.10.6/4.7.4
conda activate smartsim-dev
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/aeshao/dev/SmartRedis/install/lib

python call_MOM6.py
