#!/bin/bash
#SBATCH --job-name=seq96_label48_p96_pati7_epoch1_des-first_test_after_change_en_de_to_trans
#SBATCH --output=work_output_folder/job_output_%j.txt
#SBATCH --error=work_output_folder/job_error_%j.txt
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00

module load python/3.9.0
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt

# Use the next arguments for the output file and additional Python script parameters
PYTHON_OUTPUT_FILE="$1"
shift  # Shift the arguments to access the next ones for the Python script

# Run the Python script with the remaining arguments and redirect its output
python -u run.py "$@" > "$PYTHON_OUTPUT_FILE"
