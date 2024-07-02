import os
from tqdm import tqdm
import argparse

from icecream import ic
import pandas as pd
import evaluate
from rouge import Rouge
from collections import defaultdict
from sacrebleu.metrics import BLEU, TER
import sacrebleu

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


        # ref = ref.replace(intro_map[ln], '')
        # mrefs[ln][dm].append(ref)

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
        # ref = ref.replace(intro_map[ln], '').strip()
        # pred = pred.replace(intro_map[ln], '').strip()

        # if len(pred) == 0:
        #     pred = 'nothing'

        # if len(ref) != 0:
        mrefs[ln][dm].append(ref)
        mpreds[ln][dm].append(pred)

    ic('here')

    rscores = defaultdict(lambda: defaultdict(float))
    mscores = defaultdict(lambda: defaultdict(float))
    cscores = defaultdict(lambda: defaultdict(float))
    bscores = defaultdict(lambda: defaultdict(float))
    tscores = defaultdict(lambda: defaultdict(float))
    escores = defaultdict(lambda: defaultdict(float))
    bertscores = defaultdict(lambda: defaultdict(float))
    # xlmscores = defaultdict(lambda: defaultdict(float))
    # dbertscores = defaultdict(lambda: defaultdict(float))
    # mt5scores = defaultdict(lambda: defaultdict(float))
    rouge = Rouge()
    meteor = evaluate.load('meteor')
    chrf = evaluate.load('chrf')
    bleu = BLEU()
    ter = TER()
    em = evaluate.load('exact_match')
    bert = evaluate.load('bertscore')

    for ln in tqdm(mrefs.keys(), desc='lang'):
        for dm in tqdm(mrefs[ln].keys(), desc='dom'):
            all_preds = mpreds[ln][dm]
            all_refs = mrefs[ln][dm]

            m_all = []
            c_all = []
            e_all = []
            bert_all = []
            xlm_all = []
            dbert_all = []
            mt5_all = []
            b_all = []


            for p, r in zip(all_preds, all_refs):
                results = meteor.compute(predictions=[p], references=[r])
                m_all.append(results['meteor'])

                results = chrf.compute(predictions=[p], references=[r])
                c_all.append(results['score'])

                results = sacrebleu.sentence_bleu(p, [r], use_effective_order=True).score
                b_all.append(results)

                # results = bert.compute(predictions=[p], references=[r], model_type="bert-base-multilingual-cased")
                # bert_all.append(results['f1'][0])

                results = bert.compute(predictions=[p], references=[r], model_type="xlm-mlm-100-1280")
                bert_all.append(results['f1'][0])

                # results = bert.compute(predictions=[p], references=[r], model_type="distilbert-base-multilingual-cased")
                # bert_all.append(results['f1'][0])

                # results = bert.compute(predictions=[p], references=[r], model_type="google/mt5-small")
                # bert_all.append(results['f1'][0])
                # results = bleu.compute(predictions=[p], references=[[r]])
                # b_all.append(results['bleu'])

                # x = p.split(' ')
                # y = r.split(' ')

                # if len(x) > len(y):
                #     y = y + ['<pad>']*(len(x) - len(y))
                # elif len(y) > len(x):
                #     x = x + ['<pad>']*(len(y) - len(x))
                # results = em.compute(predictions=x, references=y)
                # e_all.append(round(results['exact_match'], 2))

            scores = rouge.get_scores(all_preds, all_refs, avg=True)
            rscores[ln][dm] = scores['rouge-2']['f']
            mscores[ln][dm] = sum(m_all)/len(m_all)
            cscores[ln][dm] = sum(c_all)/len(c_all)
            # bscores[ln][dm] = sacrebleu.sentence_bleu(all_preds, all_refs, use_effective_order=True).score
            bscores[ln][dm] = sum(b_all)/len(b_all)
            # tscores[ln][dm] = ter.corpus_score(all_preds, [all_refs]).score
            # escores[ln][dm] = sum(e_all)/len(e_all)
            bertscores[ln][dm] = sum(bert_all)/len(bert_all)
            # xlmscores[ln][dm] = sum(xlm_all)/len(xlm_all)
            # dbertscores[ln][dm] = sum(dbert_all)/len(dbert_all)
            # mt5scores[ln][dm] = sum(mt5_all)/len(mt5_all)


    if not os.path.exists(output_path):
        os.mkdir(output_path)

    df = pd.DataFrame.from_dict(rscores)
    df.to_csv(f'{output_path}/rouge.csv')

    df = pd.DataFrame.from_dict(mscores)
    df.to_csv(f'{output_path}/meteor.csv')

    df = pd.DataFrame.from_dict(cscores)
    df.to_csv(f'{output_path}/chrf.csv')

    df = pd.DataFrame.from_dict(bscores)
    df.to_csv(f'{output_path}/bleu.csv')

    # df = pd.DataFrame.from_dict(tscores)
    # df.to_csv(f'{output_path}/ter.csv')

    # df = pd.DataFrame.from_dict(escores)
    # df.to_csv(f'{output_path}/em.csv')

    df = pd.DataFrame.from_dict(bertscores)
    df.to_csv(f'{output_path}/bert.csv')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path')
    parser.add_argument('--output_path')

    args = parser.parse_args()
    main(args)
