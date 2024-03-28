# Author: ray
# Date: 3/29/24

import unittest

from automate_experiments_fed import store_meta_info


class MyTestCase(unittest.TestCase):
    def test_something(self):
        root_path = "./dataset/ETT-small/"
        data_path = "ETTm2.csv"

        model_meta_info = {
            "root_path": root_path,
            "data_path": data_path, }
        store_meta_info(model_meta_info)


if __name__ == '__main__':
    unittest.main()
