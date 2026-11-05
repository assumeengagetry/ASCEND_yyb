"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
CREATED:  2020-12-17 10:12:13
"""
import sys
import numpy as np


def data_compare(file1, file3):
    """
    Verify that the data are the same
    """
    input1 = np.fromfile(file1, dtype=np.float16)
    print(input1)
    golden = np.sinh(input1)
    output = np.fromfile(file3, dtype=np.float16)
    print(output)
    print("-------------golden is :")
    print(golden)

    different_element_results = np.isclose(
        output, golden,
        rtol=1e-3,
        atol=1e-8,
        equal_nan=True)
    different_element_indexes = np.where(
        different_element_results != np.array((True,)))[0]
    return 0 if different_element_indexes.size == 0 else 1


if __name__ == '__main__':
    intput_file1 = sys.argv[1]
    output_file = sys.argv[2]
    cmp_result = data_compare(intput_file1, output_file)

    if cmp_result == 0:
        sys.exit(0)
    else:
        sys.exit(1)