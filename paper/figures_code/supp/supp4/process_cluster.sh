import os, shutil, sys

home_dir = '/home/gantonov/projects/exploring_replay/'
work_dir = os.path.join(home_dir, 'code')
logs_dir = os.path.join(home_dir, 'cluster', 'logs')

sys.path.append(work_dir)

job_name = 'trees.sh'
job_file = os.path.join(os.getcwd(), job_name)

with open(job_file, 'w') as fh:
    fh.writelines("#!/bin/bash -l\n")
    fh.writelines("#SBATCH -J trees.sh\n")
    fh.writelines("#SBATCH -o " + os.path.join(logs_dir, 'job.out.%j') + "\n")
    fh.writelines("#SBATCH -e " + os.path.join(logs_dir, 'job.err.%j') + "\n")
    fh.writelines("#SBATCH --exclusive=user\n")
    fh.writelines("#SBATCH -D " + work_dir + "\n")
    fh.writelines("#SBATCH --nodes=1\n")
    fh.writelines("#SBATCH --ntasks-per-node=1\n")
    fh.writelines("#SBATCH --cpus-per-task=1\n")
    fh.writelines("#SBATCH --mem-per-cpu=200\n")
    fh.writelines("#SBATCH --time=3-24:00:00\n")

    fh.writelines("module purge\n")
    fh.writelines("conda activate dist\n")

    fh.writelines("srun python ./supp_4_process.py")

os.system("sbatch %s"%job_file)
