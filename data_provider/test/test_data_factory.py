import os
import unittest

from data_provider.data_loader import Dataset_ETT_hour
from utils.commen import get_proje_root_path


class MyTestCaseData(unittest.TestCase):
    def test_dataloader(self):
        proj_root = get_proje_root_path()
        data_folder = os.path.join(proj_root, 'dataset/ETT-small')
        print(f"data_folder: {data_folder}")
        deh = Dataset_ETT_hour(root_path=data_folder)
        print(f"length of deh: {len(deh)}")


if __name__ == '__main__':
    unittest.main()
