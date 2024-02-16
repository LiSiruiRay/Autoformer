# Author: ray
# Date: 2/16/24
# Description:

import json
import subprocess
import argparse

# Set up command-line argument parsing
parser = argparse.ArgumentParser(description='Run experiments with specified configurations.')
parser.add_argument('-config', type=str, help='Path to the experiment configuration file', default='experiments.json')
args = parser.parse_args()

# Load experiment configurations from the specified JSON file
with open(args.config, 'r') as file:
    experiments = json.load(file)

# Loop over the experiments and run each one
# for exp in experiments:
#     job_name = f"autoformer_{exp['model_id']}"
#     output_file = f"output_{exp['model_id']}.txt"
#
#     command = [
#         'sbatch', 'run_experiment.sh', job_name,
#         '--model_id', exp['model_id'],
#         '--seq_len', exp['seq_len'],
#         '--label_len', exp['label_len'],
#         '--pred_len', exp['pred_len'],
#         # Add other parameters here
#     ]
#     command_str = ' '.join(command) + f" > {output_file}"
#
#     subprocess.run(command_str, shell=True)
counter = 0
for exp in experiments:
    job_name = f"autoformer_{exp['model_id']}"
    python_output_file = f"python_output_{exp['model_id']}.txt"

    # Read the content of the shell script
    with open('test.sh', 'r') as file:
        script_content = file.read()

    # Replace placeholders with actual values
    script_content = script_content.replace('%%GPUS%%', str(exp['gpus'])).replace('%%TIME%%', exp['time'])

    # Write the modified script to a temporary file
    temp_script_name = f'temp_test_{counter}.sh'
    counter += 1
    with open(temp_script_name, 'w') as file:
        file.write(script_content)

    command = [
        'sbatch', temp_script_name, python_output_file,
        '--model_id', exp['model_id'],
        '--seq_len', exp['seq_len'],
        '--label_len', exp['label_len'],
        '--pred_len', exp['pred_len'],
        # Add other parameters
    ]

    # Execute the command
    command_str = ' '.join(command) + f" > sbatch_output_{exp['model_id']}.txt"
    #
    print(command_str)
    # subprocess.run(command_str, shell=True)

    # Optionally, delete the temporary script if you don't need it anymore
    # os.remove(temp_script_name)
