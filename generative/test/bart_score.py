import argparse

from icecream import ic
import pandas as pd
from rouge import Rouge
from collections import defaultdict

def main(args):

    data_path = args.data_path
    output_path = args.output_path

    df = pd.read_csv(data_path)
    lang = df['lang']
    dom = df['domain']
    refs = df['ref_text']
    preds = df['pred_text']

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

    allkeys = set(list(languages_map.keys()))
    allvals = set(list(languages_map.values()))


    mrefs = defaultdict(lambda: defaultdict(list))
    mpreds = defaultdict(lambda: defaultdict(list))

    for i in range(len(df)):
        ref = refs[i]
        pred = preds[i]
        ln = lang[i]
        dm = dom[i]

        maybe = ref[:2]
        if maybe in allkeys:
            ref = ref[5:]

        ref = ref.replace('<SEP>', '')
        ref = ref.strip()
        mrefs[ln][dm].append(ref)

        maybe = pred[:2]
        if maybe in allkeys:
            pred = pred[5:]
        pred = pred.replace('<SEP>', '')
        pred = pred.strip()
        mpreds[ln][dm].append(pred)

    rscores = defaultdict(lambda: defaultdict(float))
    rouge = Rouge()

    for ln in mrefs.keys():
        for dm in mrefs[ln].keys():
            all_preds = mpreds[ln][dm]
            all_refs = mrefs[ln][dm]
            scores = rouge.get_scores(all_preds, all_refs, avg=True)
            rscores[ln][dm] = scores['rouge-1']['f']

    df = pd.DataFrame.from_dict(rscores)
    df.to_csv(output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path')
    parser.add_argument('--output_path')

    args = parser.parse_args()
    main(args)
