#!/bin/bash
#SBATCH --job-name=submit_nao_skill_process
#SBATCH --partition=short-serial
#SBATCH --time=1200:00
#SBATCH -o /home/users/benhutch/energy-met-corr/logs/submit_nao_skill_process-%A_%a.out
#SBATCH -e /home/users/benhutch/energy-met-corr/logs/submit_nao_skill_process-%A_%a.err

# Set up the usage messages
usage="Usage: sbatch submit_nao_skill_process.bash"

# Check the number of CLI arguments
if [ "$#" -ne 0 ]; then
    echo "Illegal number of parameters"
    echo $usage
    exit 1
fi

module load jaspy

# Set up the process script
process_script="/home/users/benhutch/energy-met-corr/nao_skill_process.py"

# Run the script
python ${process_script}

# End of file
echo "End of file"