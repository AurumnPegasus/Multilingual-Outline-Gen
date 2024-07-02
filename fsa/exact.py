import argparse
from icecream import ic
import json
import os
from tqdm import tqdm
import pandas as pd
from collections import defaultdict

def main(args):

    data_path = args.data_path
    sent_path = args.sent_path
    output_path = args.output_path

    df = pd.read_csv(sent_path)
    temp_dict = df.to_dict()
    fsa_dict = defaultdict(lambda: defaultdict(str))

    dom_dict = temp_dict['Unnamed: 0']
    temp_dict.pop('Unnamed: 0')

    langs = df.columns.tolist()[1:]
    doms = list(dom_dict.values())

    for ln in temp_dict.keys():
        for dm in temp_dict[ln].keys():
            fsa_dict[ln][dom_dict[dm]] = temp_dict[ln][dm]

    fp = open(data_path, 'r')
    df = [json.loads(line, strict=False) for line in fp.readlines()]

    rall = defaultdict(lambda: defaultdict(list))

    for row in tqdm(df, desc='df'):
        sections = row['article']['sections']
        titles = []
        for sec in sections:
            titles.append(sec['title'])

        lang, dom = row['language'], row['domain']

        refs = ' '.join(titles)
        preds = fsa_dict[lang][dom]

        refs = refs.strip()
        preds = preds.strip()

        i = 0 if refs != preds else 1
        rall[lang][dom].append(i)


    score = defaultdict(lambda: defaultdict(float))
    for ln in langs:
        for dm in doms:
            score[ln][dm] = sum(rall[ln][dm])/len(rall[ln][dm])

    df = pd.DataFrame.from_dict(score)
    df.to_csv(output_path)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path')
    parser.add_argument('--sent_path')
    parser.add_argument('--output_path')

    args = parser.parse_args()
    main(args)
