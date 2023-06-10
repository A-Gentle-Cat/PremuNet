from typing import Union

import numpy as np
import torch


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        if len(args) == 1 and isinstance(args[0], list):
            self.data = [a + float(b) for a, b in zip(self.data, args[0])]
        else:
            self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)