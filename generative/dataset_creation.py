import os
import json
import argparse
import pandas as pd

from tqdm import tqdm
from rouge import Rouge
from icecream import ic
from random import shuffle
from collections import defaultdict

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

def writeFile(path, data):
    f = open(path, 'w')
    for dic in data:
        f.write(json.dumps(dic, ensure_ascii=False))
        f.write('\n')


def main(args):

    data_path = args.data_path
    output_path = args.output_path
    train = []
    test = []
    val = []

    ln_names = getFileNames(data_path)

    for ln in tqdm(ln_names, desc='languages'):
        ln_path = f'{data_path}/{ln}'
        dom_names = getFileNames(ln_path)

        for dom in tqdm(dom_names, desc='domains'):
            dom_path = f'{ln_path}/{dom}'
            dom = dom[3:-5]
            dataset = getFileData(dom_path)
            xdataset = []
            for article in dataset:
                xdataset.append({
                    'article': article,
                    'language': ln,
                    'domain': dom
                })

            shuffle(xdataset)

            train.extend(xdataset[:int(0.8*len(xdataset))])
            val.extend(xdataset[int(0.8*len(xdataset)):int(0.9*len(xdataset))])
            test.extend(xdataset[int(0.9*len(xdataset)):])

    shuffle(train)
    shuffle(test)
    shuffle(val)

    output_path = f'{output_path}'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    writeFile(f'{output_path}/train.json', train)
    writeFile(f'{output_path}/test.json', test)
    writeFile(f'{output_path}/val.json', val)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Input parameters for stat calculations')
    parser.add_argument('--data_path', help='path to folder containing all data files')
    parser.add_argument('--output_path', help='path to output csv file')

    args = parser.parse_args()
    main(args)
