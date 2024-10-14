#!/bin/bash -l
#SBATCH -J process_trees.sh
#SBATCH -o /home/gantonov/projects/exploring_replay/paper/cluster/logs/job.out.%j
#SBATCH -e /home/gantonov/projects/exploring_replay/paper/cluster/logs/job.err.%j
#SBATCH --exclusive=user
#SBATCH -D /home/gantonov/projects/exploring_replay/paper/figures_code/supp/supp4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=200
#SBATCH --time=3-24:00:00
module purge
conda activate replay
srun python ./supp_4_process.py