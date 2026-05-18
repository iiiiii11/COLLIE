import datetime
from functools import wraps
import os
import re

import numpy as np


def Disabled(func):
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        raise Exception(f"function {func.__name__} is disabled")

    return wrapped_func


def save_cmd_args(args, save_dir):
    import os
    import sys
    import yaml

    with open(os.path.join(save_dir, "args.yaml"), "w", encoding="utf-8") as f:

        yaml.dump(vars(args), f, default_flow_style=False, allow_unicode=True)

    cmd = "python " + " ".join(arg for arg in sys.argv)

    with open(os.path.join(save_dir, "cmd.txt"), "w", encoding="utf-8") as f:
        f.write(cmd + "\n")


def getDataTimeString():
    return datetime.datetime.now().strftime('%Y%m%d-%H%M%S')[2:]


def get_file(dir_path, prefix, suffix, index):

    if not os.path.exists(dir_path):
        return None

    files = os.listdir(dir_path)

    matching_files = []

    pattern = re.compile(f'^{re.escape(prefix)}(\\d+){re.escape(suffix)}$')

    for file in files:
        match = pattern.match(file)
        if match:
            file_index = int(match.group(1))
            matching_files.append((file_index, file))

    if not matching_files:
        print(f"no matching files for {prefix}-?-{suffix}")
        return None

    if index is not None:

        for file_index, filename in matching_files:
            if file_index == index:
                return os.path.join(dir_path, filename)
        return None
    else:

        max_index_file = max(matching_files, key=lambda x: x[0])
        return os.path.join(dir_path, max_index_file[1])


class EasyDict:
    def __init__(self, data: dict):
        self.__dict__.update(data)

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, item, value):
        self.__dict__[item] = value


def grid_sample_flat(a, b, c, d, e, f):

    x_1d = np.arange(a, a + c + 1e-12, e)
    y_1d = np.arange(b, b + d + 1e-12, f)
    X, Y = np.meshgrid(x_1d, y_1d)

    x = X.ravel()
    y = Y.ravel()
    loc = np.stack([x, y], axis=-1)
    return loc


import psutil


def kill_all_children():
    current_process = psutil.Process()
    children = current_process.children(recursive=True)
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    gone, alive = psutil.wait_procs(children, timeout=3)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
