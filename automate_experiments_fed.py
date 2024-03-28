# Author: ray
# Date: 2/16/24
# Description:

import json
import os
import subprocess
import argparse
from datetime import datetime
import pytz

def get_time_string():
    # Define the timezone for Abu Dhabi
    abu_dhabi_tz = pytz.timezone('Asia/Dubai')

    # Get the current time in Abu Dhabi timezone
    now_abu_dhabi = datetime.now(abu_dhabi_tz)

    # Format the time as a string without spaces, '-', or '_'
    time_string_id = now_abu_dhabi.strftime('%Y%m%d@%Hh%Mm%Ss')

    return time_string_id

def store_dataset_info(root_path: str, data_path: str):
    pass

def store_model_meta_info()

def store_meta_info(meta_info: dict):
    meta_info_folder = "meta_info"
    if not os.path.exists(meta_info_folder):
        os.makedirs(meta_info_folder)
        print(f"Folder '{meta_info_folder}' created.")
    else:
        print(f"Folder '{meta_info_folder}' already exists.")

    root_path = meta_info["root_path"]
    data_path = meta_info["data_path"]
    store_dataset_info(root_path=root_path, data_path=data_path)
    pass


def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Run experiments with specified configurations.')
    parser.add_argument('-config', type=str, help='Path to the experiment configuration file',
                        default='experiments.json')
    args = parser.parse_args()

    # Load experiment configurations from the specified JSON file
    with open(args.config, 'r') as file:
        experiments = json.load(file)

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
        factor = exp['factor'] if 'factor' in exp else 1
        enc_in = exp['enc_in'] if 'enc_in' in exp else 7
        dec_in = exp['dec_in'] if 'dec_in' in exp else 7
        c_out = exp['c_out'] if 'c_out' in exp else 7
        des = exp['des'] if 'des' in exp else 'test'
        freq = exp['freq'] if 'freq' in exp else 'h'
        itr = exp['itr'] if 'itr' in exp else 3
        task_id = exp['task_id'] if 'task_id' in exp else "default_task"
        d_model = exp['d_model'] if 'd_model' in exp else 512
        version = exp['version'] if 'version' in exp else 'Fourier'
        mode_select = exp['mode_select'] if 'mode_select' in exp else 'random'
        modes = exp['modes'] if 'modes' in exp else 64
        L = exp['L'] if 'L' in exp else 3
        base = exp['base'] if 'base' in exp else 'legendre'
        cross_activation = exp['cross_activation'] if 'cross_activation' in exp else 'tanh'
        target = exp['target'] if 'target' in exp else 'OT'

        detail_freq = exp['detail_freq'] if 'detail_freq' in exp else 'h'

        n_heads = exp['n_heads'] if 'n_heads' in exp else 8
        d_ff = exp['d_ff'] if 'd_ff' in exp else 2048
        moving_avg = exp['moving_avg'] if 'moving_avg' in exp else [24]
        dropout = exp['dropout'] if 'dropout' in exp else 0.05
        embed = exp['embed'] if 'embed' in exp else 'timeF'
        activation = exp['activation'] if 'activation' in exp else 'gelu'

        # action
        output_attention = exp['output_attention'] if 'output_attention' in exp else False
        distil = exp['distil'] if 'distil' in exp else True
        do_predict = exp['do_predict'] if 'do_predict' in exp else False
        use_amp = exp['use_amp'] if 'use_amp' in exp else False
        use_multi_gpu = exp['use_multi_gpu'] if 'use_multi_gpu' in exp else False

        work_output_folder = exp['work_output_folder'] if 'work_output_folder' in exp else "work_output_folder"

        description = exp['description'] if 'description' in exp else ''

        # model_id = f"seq{seq_len}_label{label_len}_p{pred_len}_pati{patience}_epoch{train_epochs}"
        model_id = get_time_string()

        # job_name = f"autoformer_{model_id}"
        python_output_file = f"{work_output_folder}/python_output_{model_id}.txt"

        model_name = '{}_{}_{}_modes{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            task_id,
            model,
            mode_select,
            modes,
            data,
            features,
            seq_len,
            label_len,
            pred_len,
            d_model,
            n_heads,
            e_layers,
            d_layers,
            d_ff,
            factor,
            embed,
            distil,
            des,
            0)

        # Read the content of the shell script
        with open('experiment.sh', 'r') as file:
            script_content = file.read()

        # Replace placeholders with actual values
        script_content = (script_content.replace('%%GPUS%%', str(exp['gpus']) if "gpus" in exp else '3')
                          .replace('%%TIME%%', exp['time'] if "time" in exp else "4:00:00")
                          .replace('%%JOBNAME%%', model_id)
                          .replace('%%OUTPUT%%', f"{work_output_folder}/job_output_%j.txt")
                          .replace('%%ERROR%%', f"{work_output_folder}/job_error_%j.txt"))

        # Write the modified script to a temporary file
        temp_script_name = f'temp.sh'
        with open(temp_script_name, 'w') as file:
            file.write(script_content)

        command = [
            'sbatch', temp_script_name, python_output_file,
            # '--model_id', model_id,
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
            '--itr', itr,
            '--task_id', task_id,

            # added
            '--version', version,
            '--mode_select', mode_select,
            '--modes', modes,
            '--L', L,
            '--base', base,
            '--cross_activation', cross_activation,
            '--target', target,
            '--d_model', d_model,

            # Actions
            *(['--output_attention'] if output_attention else []),
            *(['--distil'] if distil else []),
            *(['--do_predict'] if do_predict else []),
            *(['--use_amp'] if use_amp else []),
            *(['--use_multi_gpu'] if use_multi_gpu else []),
        ]

        command = [str(i) for i in command]

        model_meta_info = {
            "model_name": model_name,
            "seq_len": seq_len,
            "label_len": label_len,
            "pred_len": pred_len,
            "patience": patience,
            "train_epochs": train_epochs,
            "root_path": root_path,
            "data_path": data_path,
            "is_training": is_training,
            "features": features,
            "model": model,
            "data": data,
            "e_layers": e_layers,
            "d_layers": d_layers,
            "factor": factor,
            "enc_in": enc_in,
            "dec_in": dec_in,
            "c_out": c_out,
            "des": des,
            "freq": freq,
            "itr": itr,
            "task_id": task_id,
            "d_model": d_model,
            "version": version,
            "mode_select": mode_select,
            "modes": modes,
            "L": L,
            "base": base,
            "cross_activation": cross_activation,
            "target": target,
            "detail_freq": detail_freq,
            "n_heads": n_heads,
            "d_ff": d_ff,
            "moving_avg": moving_avg,
            "dropout": dropout,
            "embed": embed,
            "activation": activation,
            "output_attention": output_attention,
            "distil": distil,
            "do_predict": do_predict,
            "use_amp": use_amp,
            "use_multi_gpu": use_multi_gpu,
            "work_output_folder": work_output_folder,
            "description": description,
            "model_id": model_id,
            "python_output_file": python_output_file,
            "script_content": script_content,
            "command": command
        }


        if not os.path.exists(work_output_folder):
            os.makedirs(work_output_folder)
            print(f"Folder '{work_output_folder}' created.")
        else:
            print(f"Folder '{work_output_folder}' already exists.")
        # Execute the command
        command_str = ' '.join(command) + f" > {work_output_folder}/sbatch_output_{model_id}.txt"
        print(command_str)

        subprocess.run(command_str, shell=True)

        # Optionally, delete the temporary script if you don't need it anymore
        # os.remove(temp_script_name)


if __name__ == '__main__':
    main()
