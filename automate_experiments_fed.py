# Author: ray
# Date: 2/16/24
# Description:
import csv
import json
import os
import subprocess
import argparse
from datetime import datetime
import pytz

import hashlib
import shutil


def generate_id_from_csv(file_path):
    # Initialize an empty string to store the CSV content
    content = ""

    # Open the CSV file and read its content
    with open(file_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            # Convert the row to a string and append it to the content
            content += ','.join(row) + '\n'

    # Convert the content to bytes
    content_bytes = content.encode('utf-8')

    # Create a SHA-256 hash object
    hash_object = hashlib.sha256(content_bytes)

    # Generate the hash
    hex_dig = hash_object.hexdigest()

    return hex_dig


def get_time_string():
    # Define the timezone for Abu Dhabi
    abu_dhabi_tz = pytz.timezone('Asia/Dubai')

    # Get the current time in Abu Dhabi timezone
    now_abu_dhabi = datetime.now(abu_dhabi_tz)

    # Format the time as a string without spaces, '-', or '_'
    time_string_id = now_abu_dhabi.strftime('%Y%m%d@%Hh%Mm%Ss')

    return time_string_id


def store_dataset_info(root_path: str, data_path: str, dataset_id: str, meta_info_folder="meta_info"):
    meta_info_dataset_path = os.path.join(meta_info_folder, "datasets_info")
    if not os.path.exists(meta_info_dataset_path):
        os.makedirs(meta_info_dataset_path)
        print(f"Folder '{meta_info_dataset_path}' created.")
    else:
        print(f"Folder '{meta_info_dataset_path}' already exists.")

    destination_path = os.path.join(meta_info_dataset_path, f"{dataset_id}.csv")

    if os.path.exists(destination_path):
        print(f"dataset existed")
        return

    source_path = os.path.join(root_path, data_path)
    shutil.copy(source_path, destination_path)


def store_model_meta_info(meta_info: dict, meta_info_folder="meta_info"):
    meta_info_generated_ts = get_time_string()
    task_id = meta_info["task_id"]
    meta_data_file_folder_path = os.path.join(meta_info_folder, "model_meta_info")

    if not os.path.exists(meta_data_file_folder_path):
        os.makedirs(meta_data_file_folder_path)
        print(f"Folder '{meta_data_file_folder_path}' created.")
    else:
        print(f"Folder '{meta_data_file_folder_path}' already exists.")

    meta_data_file_path = os.path.join(meta_data_file_folder_path, f"{task_id}_{meta_info_generated_ts}.json")
    with open(meta_data_file_path, "w") as f:
        json.dump(meta_info, f, indent=4)


def store_meta_info(meta_info: dict):
    meta_info_folder = "meta_info"
    if not os.path.exists(meta_info_folder):
        os.makedirs(meta_info_folder)
        print(f"Folder '{meta_info_folder}' created.")
    else:
        print(f"Folder '{meta_info_folder}' already exists.")

    root_path = meta_info["root_path"]
    data_path = meta_info["data_path"]

    dataset_path = os.path.join(root_path, data_path)
    dataset_id = generate_id_from_csv(dataset_path)
    meta_info["dataset_id"] = dataset_id

    store_dataset_info(root_path=root_path, data_path=data_path, dataset_id=dataset_id)
    store_model_meta_info(meta_info=meta_info)


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
        task_id = f"{task_id}_{get_time_string()}"

        # job_name = f"autoformer_{model_id}"
        python_output_file = f"{work_output_folder}/python_output_{task_id}.txt"

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
                          .replace('%%JOBNAME%%', task_id)
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
            # "model_id": model_id,
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
