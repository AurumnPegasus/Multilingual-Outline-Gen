import argparse
import pandas as pd
from icecream import ic
from collections import defaultdict


def getdict(df, nonFSA=True):
    temp = df.to_dict()
    df_dict = defaultdict(lambda: defaultdict(str))
    dom_dict = temp['Unnamed: 0']
    temp.pop('Unnamed: 0')

    for ln in temp.keys():
        for dm in temp[ln].keys():
            if nonFSA:
                df_dict[ln[:2]][dom_dict[dm]] = temp[ln][dm]
            else:
                df_dict[ln][dom_dict[dm]] = temp[ln][dm]

    return df_dict


def main(args):

    # mbart_path = args.mbart_path
    mt5_path = args.mt5_path
    fsa_path = args.fsa_path
    output_path = args.output_path

    # mbart = pd.read_csv(mbart_path)
    mt5 = pd.read_csv(mt5_path)
    fsa = pd.read_csv(fsa_path)

    temp  = fsa.to_dict()
    dom_dict = temp['Unnamed: 0']
    temp.pop('Unnamed: 0')
    langs = fsa.columns.tolist()[1:]
    doms = list(dom_dict.values())

    fsa_dict = getdict(fsa, False)
    # mbart_dict = getdict(mbart)
    mt5_dict = getdict(mt5)

    ans_dict = defaultdict(lambda: defaultdict(str))
    score_dict = defaultdict(lambda: defaultdict(float))

    for ln in langs:
        for dm in doms:
            # f, b, t = fsa_dict[ln][dm], mbart_dict[ln][dm], mt5_dict[ln][dm]
            f, t = fsa_dict[ln][dm], mt5_dict[ln][dm]

            if f >= t:
                ans_dict[ln][dm] = 'fsa'
                score_dict[ln][dm] = f
            else:
                ans_dict[ln][dm] = 'mt5'
                score_dict[ln][dm] = t

    df = pd.DataFrame.from_dict(ans_dict)
    df.to_csv('./choices.csv')

    # df = pd.DataFrame.from_dict(score_dict)
    # df.to_csv(output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument('--mbart_path')
    parser.add_argument('--mt5_path')
    parser.add_argument('--fsa_path')
    parser.add_argument('--output_path')

    args = parser.parse_args()
    main(args)
