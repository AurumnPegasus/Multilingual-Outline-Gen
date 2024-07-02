import os
import argparse

from icecream import ic
import pandas as pd
import evaluate
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

    intro_map = {
            'bn_IN': 'ভূমিকা',
            'en_XX': 'Introduction',
            'hi_IN': 'परिचय',
            'kn_IN': 'ಪರಿಚಯ',
            'ml_IN': 'ആമുഖം',
            'mr_IN': 'परिचय',
            'or_IN': 'ପରିଚୟ',
            'pa_IN': 'ਜਾਣ-ਪਛਾਣ',
            'ta_IN': 'அறிமுகம்',
            'te_IN': 'పరిచయం'
        }


    mrefs = defaultdict(lambda: defaultdict(list))
    mpreds = defaultdict(lambda: defaultdict(list))

    count = 0

    for i in range(len(df)):
        ref = refs[i]
        pred = preds[i]
        ln = lang[i]
        dm = dom[i]

        word = ref[:5]
        if word in languages_map.values():
            ref = ref.replace(word, '')

        word = ref[:2]
        if word in languages_map.keys():
            ref = ref.replace(word, '')

        ref = ref.replace('<SEP>', '')
        ref = ref.strip()
        mrefs[ln][dm].append(ref)

        try:
            word = pred[:5]
            if word in languages_map.values():
                pred = pred.replace(word, '')
            word = pred[:2]
            if word in languages_map.keys():
                pred = pred.replace(word, '')
        except:
            pass

        try:
            pred = pred.replace('<SEP>', '')
            pred = pred.strip()
        except:
            pred = intro_map[ln]

        ref = ref
        pred = pred
        mpreds[ln][dm].append(pred)

        if pred.strip() in intro_map[ln]:
            count += 1
    ic(len(df), count)

    # rscores = defaultdict(lambda: defaultdict(float))
    # mscores = defaultdict(lambda: defaultdict(float))
    # cscores = defaultdict(lambda: defaultdict(float))
    # rouge = Rouge()
    # meteor = evaluate.load('meteor')
    # chrf = evaluate.load('chrf')

    # for ln in mrefs.keys():
    #     for dm in mrefs[ln].keys():
    #         all_preds = mpreds[ln][dm]
    #         all_refs = mrefs[ln][dm]

    #         m_all = []
    #         c_all = []

    #         for p, r in zip(all_preds, all_refs):
    #             results = meteor.compute(predictions=[p], references=[r])
    #             m_all.append(results['meteor'])

    #             results = chrf.compute(predictions=[p], references=[r], word_order=2)
    #             c_all.append(results['score'])

    #         scores = rouge.get_scores(all_preds, all_refs, avg=True)
    #         rscores[ln][dm] = scores['rouge-l']['f']
    #         mscores[ln][dm] = sum(m_all)/len(m_all)
    #         cscores[ln][dm] = sum(c_all)/len(c_all)

    # if not os.path.exists(output_path):
    #     os.mkdir(output_path)

    # df = pd.DataFrame.from_dict(rscores)
    # df.to_csv(f'{output_path}/rouge.csv')

    # df = pd.DataFrame.from_dict(mscores)
    # df.to_csv(f'{output_path}/meteor.csv')

    # df = pd.DataFrame.from_dict(cscores)
    # df.to_csv(f'{output_path}/chrf.csv')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path')
    parser.add_argument('--output_path')

    args = parser.parse_args()
    main(args)
