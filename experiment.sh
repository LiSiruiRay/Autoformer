#!/bin/bash
#SBATCH --job-name=%%JOBNAME%%
#SBATCH --output=%%OUTPUT%%
#SBATCH --error=%%ERROR%%
#SBATCH -p nvidia
#SBATCH --gres=gpu:%%GPUS%%
#SBATCH --time=%%TIME%%

module load python/3.9.0
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt

# Use the next arguments for the output file and additional Python script parameters
PYTHON_OUTPUT_FILE="$1"
shift  # Shift the arguments to access the next ones for the Python script

# Run the Python script with the remaining arguments and redirect its output
python -u run.py "$@" > "$PYTHON_OUTPUT_FILE"
