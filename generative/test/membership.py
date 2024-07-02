import os
import json
import argparse
# import evaluate
import pandas as pd

from tqdm import tqdm
from rouge import Rouge
from icecream import ic
from collections import defaultdict

def getFileNames(path):
    fnames = [f for f in os.listdir(path)]
    return fnames


def main(args):

    data_path = args.data_path
    ref_path = args.ref_path

    ln_names = getFileNames(ref_path)
    ref_data = []
    all_titles = set()

    languages_map = {
            'bn': 'bn_IN',
            'de': 'de_DE',
            'en': 'en_XX',
            'es': 'es_XX',
            'fr': 'fr_XX',
            'gu': 'gu_IN',
            'hi': 'hi_IN',
            'it': 'it_IT',
            'kn': 'kn_IN',
            'ml': 'ml_IN',
            'mr': 'mr_IN',
            'or': 'or_IN',
            'pa': 'pa_IN',
            'ta': 'ta_IN',
            'te': 'te_IN',
        }


    for ln in ln_names:
        fnames = getFileNames(f'{ref_path}/{ln}')

        for fn in fnames:
            fn_path = f'{ref_path}/{ln}/{fn}'

            if 'test' not in fn_path:
                continue
            f = open(fn_path)

            df = [json.loads(line, strict=False) for line in f.readlines()]
            domain = fn.split('_')[0]

            for row in df:
                sections = row['sections']
                content = ""
                for sec in sections:
                    content = content + ' ' + sec['content']
                if len(content.split(' ')) >= 50:
                    ref_data.append(row)
                    all_titles.add(languages_map[ln] + ' ' + domain+ ' ' + row['title'])

    df = pd.read_csv(data_path)
    all_gen = []
    for row in df.iterrows():
        # title = ' '.join(row[1]['input_texts'].split(' ')[2:]).strip()
        title = row[1]['input_texts']
        # ic(row)
        if title in all_titles:
            all_gen.append({
                'input_texts': row[1]['input_texts'],
                'lang': row[1]['lang'],
                'domain': row[1]['domain'],
                'ref_text': row[1]['ref_text'],
                'pred_text': row[1]['pred_text']
            })

    ic(len(all_gen), len(all_titles))

    df = pd.DataFrame(all_gen)
    df.to_csv('modified_mt5.csv')




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path')
    parser.add_argument('--ref_path')

    args = parser.parse_args()
    main(args)
