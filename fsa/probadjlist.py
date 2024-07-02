import os
import json
import random
import argparse
import evaluate
import pandas as pd

from tqdm import tqdm
from rouge import Rouge
from icecream import ic
from collections import defaultdict
from sacrebleu.metrics import BLEU
import sacrebleu

chrf = evaluate.load('chrf')
meteor = evaluate.load('meteor')
rouge = Rouge()
bleu = BLEU()
bert = evaluate.load('bertscore')

def getFileNames(path):
    fnames = [f for f in os.listdir(path)]
    return fnames

def getMatrix(df, word_level):
    matrix = defaultdict(lambda: defaultdict(int))
    total = defaultdict(int)

    for article in df:

        # getting markovian repr for all section titles in article
        prev = '<source>'
        for section in article['sections']:

            # word level FSA or title level FSA
            if word_level:
                current = section['title']
                tokens = current.split(' ')
                # ic(tokens)
                for tok in tokens:
                    if tok == '':
                        continue
                    # if tok.strip() == 'Introduction':
                    #     tok = intro_map[ln]
                    tok = tok.strip()
                    matrix[prev][tok] += 1
                    total[prev] += 1
                    prev = tok
                    # ic(tok)
            else:
                current = section['title']
                # current = current.replace('Introduction', intro_map[ln])
                matrix[prev][current] += 1
                total[prev] += 1
                prev = current

        # finally adding sink
        current = '<sink>'
        matrix[prev][current] += 1
        total[prev] += 1
        # c += 1

        # if c == 5:
        #     ic(matrix)
        #     exit(0)


    # normalising
    for key in matrix.keys():
        for word in matrix[key]:
            matrix[key][word] = matrix[key][word] / total[key]

        # sorting according to highest occurance probability
        matrix[key] = {k: v for k, v in sorted(matrix[key].items(), key=lambda item: item[1], reverse=True)}

    return matrix



# def getMatrix(df, word_level):
#     matrix = defaultdict(lambda: defaultdict(int))
#     total = defaultdict(int)

#     for article in df:

#         # getting markovian repr for all section titles in article
#         prev = '<source>'
#         for section in article['sections']:
#             current = section['title']
#             matrix[prev][current] += 1
#             total[prev] += 1
#             prev = current

#         # finally adding sink
#         current = '<sink>'
#         matrix[prev][current] += 1
#         total[prev] += 1

#     # normalising
#     for key in matrix.keys():
#         for word in matrix[key]:
#             matrix[key][word] = matrix[key][word] / total[key]

#         # sorting according to highest occurance probability
#         matrix[key] = {k: v for k, v in sorted(matrix[key].items(), key=lambda item: item[1], reverse=True)}

#     return matrix

def fsa(beam, size, matrix, num_outputs, prob):
    fsa_q = [{
            'prev': '<source>',
            'prob': 1,
            'steps': 0,
            'history': ['<source>']
        }]
    possible_outputs = []

    while len(possible_outputs) < num_outputs and len(fsa_q) > 0:

        # popping the current node
        node = fsa_q.pop(0)
        prev = node['prev']
        prob = node['prob']
        history = node['history']

        # always taking the current node
        steps = node['steps'] + 1

        # reached sink, now its a valid possible output
        if prev == '<sink>':
            possible_outputs.append((history, prob**(1/steps)))

        # most probable outputs giving previous heading
        topbeams = list(matrix[prev].items())
        indices = [i for i in range(len(topbeams))]
        probs = [p for _,p in topbeams]

        if prob:

            # sampling based on weighted probablities
            choices = []
            while len(choices) < beam and sum(probs) > 0:
                index = random.choices(indices, weights=probs, k=1)
                index = index[0]
                # ic(len(topbeams), index)
                choices.append(topbeams[index])

                # removing samples to ensure no repititions
                indices[index] = -1
                # topbeams.pop(index)
                probs[index] = 0
        else:

            # cumulative sum probablity sampling
            indices = random.choices(indices, cum_weights=probs, k=beam)
            indices = list(set(indices))
            choices = [topbeams[i] for i in indices]

        for t,p in choices:

            # either the next title is not sink, or it is sink and satisfies size constraint
            if t != '<sink>' or (t == '<sink>' and steps >= size + 1):

                # prevent cycles
                if t not in history:
                    fsa_q.append({
                        'prev': t,
                        'prob': prob*p,
                        'steps': steps,
                        'history': history + [t]
                    })

    return possible_outputs

def iterate(matrix, beam, size, num_outputs, refs, prob):
    rouge = Rouge()
    # meteor = evaluate.load('meteor')
    possible_outputs = fsa(beam, size, matrix, num_outputs, prob)
    possible_outputs = sorted(possible_outputs, key=lambda item: item[1], reverse=True)

    try:
        # generating most probable outline from fsa
        sentence = ' '.join(possible_outputs[0][0])
        sentence = sentence.replace('<source>', '')
        sentence = sentence.replace('<sink>', '')
        sentence = sentence.strip()
    except:
        return -1

    # getting predictions for rouge calculation
    preds = []
    sentence = sentence.lower()
    for _ in refs:
        preds.append(sentence)
    refs = [r.lower() for r in refs]

    m_all = []
    c_all = []
    b_all = []
    bert_all = []
    for r, p in zip(refs, preds):
        results = meteor.compute(predictions=[p], references=[r])
        m_all.append(results['meteor'])
        results = chrf.compute(predictions=[p], references=[r])
        c_all.append(results['score'])

        results = sacrebleu.sentence_bleu(p, [r], use_effective_order=True).score
        b_all.append(results)

        results = bert.compute(predictions=[p], references=[r], model_type="xlm-mlm-100-1280")
        bert_all.append(results['f1'][0])



    scores = rouge.get_scores(preds, refs, avg=True)
    # return scores['rouge-l']['f']

    return sum(m_all)/len(m_all), sum(c_all)/len(c_all), sum(b_all)/len(b_all), sum(bert_all)/len(bert_all), scores['rouge-l']['f']

def main(args):

    data_path = args.data_path
    out_path = args.output_path
    word_level = args.word_level
    num_outputs = args.num_outputs
    beams = args.beams
    beams = [int(b) for b in beams.split(',')]
    fix_size = args.fix_size
    fix_size = [int(f) for f in fix_size.split(',')]
    avg = args.avg
    prob = args.prob
    sep = args.sep

    ln_names = getFileNames(data_path)

    # grid search through beams and min size
    for beam in tqdm(beams, desc="beams"):
        for size in tqdm(fix_size, desc="sizes"):

            # score dictionary for each pair of hyperparameters
            rdic = defaultdict(lambda: defaultdict(float))
            bdic = defaultdict(lambda: defaultdict(float))
            bertdic = defaultdict(lambda: defaultdict(float))
            mdic = defaultdict(lambda: defaultdict(float))
            cdic = defaultdict(lambda: defaultdict(float))

            # through each lang and domain
            for ln in tqdm(ln_names, desc='languages'):
                ln_path = f'{data_path}/{ln}'
                dom_names = getFileNames(ln_path)

                for dom in tqdm(dom_names, desc='domains'):
                    dom_path = f'{ln_path}/{dom}'
                    if 'train' not in dom_path:
                        continue

                    dom = dom[:-11]

                    # getting main dataset for lang/dom pair
                    f = open(dom_path)
                    df = [json.loads(line, strict=False) for line in f.readlines()]

                    # constructing adjacency matrix for given dataset
                    matrix = getMatrix(df, word_level)


                    val_path = f'{ln_path}/{dom}_{sep}.json'

                    # getting main dataset for lang/dom pair
                    f = open(val_path)
                    df = [json.loads(line, strict=False) for line in f.readlines()]

                    refs = []
                    for article in df:
                        sections = article['sections']
                        outline = ''

                        for section in sections:
                            outline = f'{outline} {section["title"]}'
                        refs.append(outline)

                    # computing average over n trials to get mean readings
                    rlist = []
                    blist = []
                    bertlist = []
                    mlist = []
                    clist = []
                    for _ in tqdm(range(avg), desc='averaging'):

                        # if rscore != -1:
                        #     rlist.append(rscore)
                        mscore, cscore, bscore, bertscore, rscore = iterate(matrix, beam, size, num_outputs, refs, prob)
                        if bscore != -1:
                            blist.append(bscore)

                        if bertscore != -1:
                            bertlist.append(bertscore)

                        if mscore != -1:
                            mlist.append(mscore)

                        if cscore != -1:
                            clist.append(cscore)

                        if rscore != -1:
                            rlist.append(rscore)


                    if len(rlist) == 0:
                        rdic[ln][dom] = -1
                    else:
                        rdic[ln][dom] = sum(rlist)/len(rlist)
                    if len(bertlist) == 0:
                        bertdic[ln][dom] = -1
                    else:
                        bertdic[ln][dom] = sum(bertlist)/len(bertlist)

                    if len(blist) == 0:
                        bdic[ln][dom] = -1
                    else:
                        bdic[ln][dom] = sum(blist)/len(blist)

                    if len(mlist) == 0:
                        mdic[ln][dom] = -1
                    else:
                        mdic[ln][dom] = sum(mlist)/len(mlist)

                    if len(clist) == 0:
                        cdic[ln][dom] = -1
                    else:
                        cdic[ln][dom] = sum(clist)/len(clist)


            # converting output to csv
            out_df = pd.DataFrame.from_dict(rdic)
            out_df.to_csv(f'{out_path}/rouge_{beam}_{size}.csv')

            out_df = pd.DataFrame.from_dict(bdic)
            out_df.to_csv(f'{out_path}/bleu_{beam}_{size}.csv')

            out_df = pd.DataFrame.from_dict(bertdic)
            out_df.to_csv(f'{out_path}/bert_{beam}_{size}.csv')


            out_df = pd.DataFrame.from_dict(mdic)
            out_df.to_csv(f'{out_path}/meteor_{beam}_{size}.csv')

            out_df = pd.DataFrame.from_dict(cdic)
            out_df.to_csv(f'{out_path}/chrf_{beam}_{size}.csv')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Input parameters to construct adjacency list')
    parser.add_argument('--data_path', help='path to input json file from which we construct the transition matrix')
    parser.add_argument('--output_path', default=None, help='storing the adjacency matrix for later')
    parser.add_argument('--word_level', default=0, help='word level (1) or sec-title level (0)')
    parser.add_argument('--num_outputs', default=1, type=int, help='Number of possible outputs to print')
    parser.add_argument('--beams', default="1", type=str, help='number of branches to split into')
    parser.add_argument('--fix_size', default="0", type=str, help='fixed size the fsa must generate')
    parser.add_argument('--prob', default=0, type=int, help='type of prob sampling, 0 for cumulative, 1 for weighted')
    parser.add_argument('--avg', default=10, type=int, help='average over number of trials')
    parser.add_argument('--sep', default='val', type=str, help="test or val set")

    args = parser.parse_args()
    main(args)
