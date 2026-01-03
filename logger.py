import numpy as np
from common.utils import *
import csv
from matplotlib import pyplot as plt

class Logger():
    def __init__(self):
        self.data = {}

    def log_scalar(self, name, value):
        self.data.setdefault(name, []).append(value)

    def log_vector(self, name, vec):    # 1D vector
        a = np.asarray(vec).reshape(-1)
        for i, x in enumerate(a):
            self.data.setdefault(f"{name}_{i}", []).append(float(x))

    def save_to_csv(self, filename):
        # get all column names
        keys = list(self.data.keys())

        # find max column length
        max_len = max(len(v) for v in self.data.values())

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)

            # header
            writer.writerow(keys)

            # rows
            for i in range(max_len):
                row = []
                for k in keys:
                    col = self.data[k]
                    if i < len(col):
                        row.append(col[i])
                    else:
                        row.append(np.nan)
                writer.writerow(row)