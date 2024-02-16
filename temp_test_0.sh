#!/bin/bash
#SBATCH --job-name=autoformer_pure_sin
#SBATCH --output=my_job_output_%j.txt
#SBATCH --error=my_job_error_%j.txt
#SBATCH -p nvidia
#SBATCH --gres=gpu:3
#SBATCH --time=4:00:00

module load python/3.9.0
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt

# Use the next arguments for the output file and additional Python script parameters
PYTHON_OUTPUT_FILE="$1"
shift  # Shift the arguments to access the next ones for the Python script

# Run the Python script with the remaining arguments and redirect its output
python -u run.py "$@" > "$PYTHON_OUTPUT_FILE"
