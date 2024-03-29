# Author: ray
# Date: 3/29/24
# Description:
import copy
import json


def generate_diff_input_len():
    file_name = "meta_script_test_on_multi_run.json"
    template_file_name = "first_run_with_meta_script.json"
    with open(template_file_name, 'r') as f:
        template_exp_list = json.load(f)

    new_exp_list = []
    changing_para = "seq_len"
    example_exp_config = template_exp_list[0]
    seq_len = example_exp_config[changing_para]
    for i in range(0, 10):
        print(f"test the example: {new_exp_list}")
        example_exp_config[changing_para] = seq_len
        seq_len *= 2
        new_exp_list.append(copy.deepcopy(example_exp_config))

    with open(file_name, 'w') as f:
        json.dump(new_exp_list, f, indent=4)


if __name__ == '__main__':
    generate_diff_input_len()
