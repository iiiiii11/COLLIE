import torch.multiprocessing as mp
from run.train import run, START_METHOD
import os

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    mp.set_start_method(START_METHOD)
    run()
