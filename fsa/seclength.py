import os
import json
import argparse
import pandas as pd
import numpy as np

from tqdm import tqdm
from icecream import ic
from collections import defaultdict
import matplotlib.pyplot as plt

def getFileNames(path):
    fnames = [f for f in os.listdir(path)]
    return fnames

def getFileData(path):
    data = []
    with open(path) as f:
        try:
            for line in f:
                data.append(json.loads(line))
        except:
            pass
    return data

def main(args):

    data_path = args.data_path
    output_path = args.output_path

    ln_names = getFileNames(data_path)
    avg_len = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for ln in tqdm(ln_names, desc='languages'):
        ln_path = f'{data_path}/{ln}'
        dom_names = getFileNames(ln_path)

        for dom in tqdm(dom_names, desc='domains'):
            templist = []
            dom_path = f'{ln_path}/{dom}'

            if 'train' in dom_path:
                dom = dom[:-11]
            elif 'val' in dom_path:
                dom = dom[:-9]
            else:
                dom = dom[:-10]

            dataset = getFileData(dom_path)

            for article in dataset:
                size = len(article['sections'])
                size = size if size <= 3 else 4
                avg_len[ln][dom][size] += 1

            sizes = [1, 2, 3, 4]
            for s in sizes:
                if s not in avg_len[ln][dom]:
                    avg_len[ln][dom][s] = 0

    sizes = [1, 2, 3, 4]
    for ln in avg_len.keys():
        xaxis = np.arange(4)
        val = 0.1
        c = 1
        fig, ax = plt.subplots(layout='constrained')
        for dom in avg_len[ln].keys():
            x = [avg_len[ln][dom][s] for s in sizes]
            rects = ax.barh(xaxis + val*c, x, height=0.1, label=dom)
            ax.bar_label(rects, padding=3)
            c += 1

        ax.set_xticks(xaxis, sizes)
        ax.legend()
        fig.savefig(f'{output_path}/{ln}.png')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Input parameters for stat calculations')
    parser.add_argument('--data_path', help='path to folder containing all data files')
    parser.add_argument('--output_path', help='path to output csv file')

    args = parser.parse_args()
    main(args)
