#!/bin/bash
#SBATCH --job-name=latexmlc_mugat
#SBATCH --account=tesi_czaccagnino
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --container-mounts /PATH/TO/SCRIPTS:/scripts,/PATH/TO/DATA:/data,/etc/slurm/task_prolog:/etc/slurm/task_prolog
#SBATCH --container-image carminezacc/latexml-parallel


cd /data/tex

parallel -j 16 bash /scripts/convert-one-to-xml.sh {} ::: */

