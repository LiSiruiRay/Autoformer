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
for exp in experiments:
    seq_len = exp['seq_len'] if 'seq_len' in exp else 96
    label_len = exp['label_len'] if 'label_len' in exp else 96
    pred_len = exp['pred_len'] if 'pred_len' in exp else 96
    patience = exp['patience'] if 'patience' in exp else 7
    train_epochs = exp['train_epochs'] if 'train_epochs' in exp else 10
    root_path = exp['root_path'] if 'root_path' in exp else "./dataset/ETT-small/"
    data_path = exp['data_path'] if 'data_path' in exp else "ETTm2.csv"

    is_training = exp['is_training'] if 'is_training' in exp else 1
    features = exp['features'] if 'features' in exp else 'S'
    model = exp['model'] if 'model' in exp else "Autoformer"
    data = exp['data'] if 'data' in exp else "ETTm2"
    e_layers = exp['e_layers'] if 'e_layers' in exp else 2
    d_layers = exp['d_layers'] if 'd_layers' in exp else 1
    factor = exp['factor'] if 'factor' in exp else 3
    enc_in = exp['enc_in'] if 'enc_in' in exp else 1
    dec_in = exp['dec_in'] if 'dec_in' in exp else 1
    c_out = exp['c_out'] if 'c_out' in exp else 1
    des = exp['des'] if 'des' in exp else 'Exp'
    freq = exp['freq'] if 'freq' in exp else 't'
    itr = exp['itr'] if 'itr' in exp else 1

    description = exp['description'] if 'description' in exp else ''

    model_id = f"s{seq_len}_l{label_len}_p{pred_len}_pa{patience}_tp{train_epochs}_des-{description}"

    job_name = f"autoformer_{model_id}"
    python_output_file = f"python_output_{model_id}.txt"

    # Read the content of the shell script
    with open('experiment.sh', 'r') as file:
        script_content = file.read()

    # Replace placeholders with actual values
    script_content = (script_content.replace('%%GPUS%%', str(exp['gpus']) if "gpus" in exp else '3')
                      .replace('%%TIME%%', exp['time'] if "time" in exp else "4:00:00"))

    # Write the modified script to a temporary file
    temp_script_name = f'temp.sh'
    with open(temp_script_name, 'w') as file:
        file.write(script_content)

    command = [
        'sbatch', temp_script_name, python_output_file,
        '--model_id', model_id,
        '--seq_len', seq_len,
        '--label_len', label_len,
        '--pred_len', pred_len,
        '--patience', patience,
        '--train_epochs', train_epochs,
        '--root_path', root_path,
        '--data_path', data_path,
        '--is_training', is_training,
        '--model', model,
        '--data', data,
        '--features', features,
        '--e_layers', e_layers,
        '--d_layers', d_layers,
        '--factor', factor,
        '--enc_in', enc_in,
        '--dec_in', dec_in,
        '--c_out', c_out,
        '--des', des,
        '--freq', freq,
        '--itr', itr
        # Add other parameters
    ]

    command = [str(i) for i in command]

    # Execute the command
    command_str = ' '.join(command) + f" > sbatch_output_{model_id}.txt"
    print(command_str)
    subprocess.run(command_str, shell=True)

    # Optionally, delete the temporary script if you don't need it anymore
    # os.remove(temp_script_name)
