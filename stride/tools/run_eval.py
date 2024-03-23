
import argparse
import multiprocessing

from tqdm.auto import tqdm
import pandas as pd

from ..db import NGramDBMulti
from ..corpus import Corpus, Entry
from ..vocab import Vocab
from ..predict import predict_multi



def init(args):
    global dbs, vocab, typ, flanking
    dbs = [
        NGramDBMulti.load(path)
        for path in args.dbs
    ]
    vocab = Vocab.load(args.vocab)
    typ = args.type
    flanking = args.flanking


def predict_one(entry: Entry):
    global dbs, vocab, typ, flanking
    p = predict_multi(entry, vocab, dbs, typ, flanking)

    # Filter to only human labels and attach the ground truth label and count.

    preds = []
    counts = entry.var_counts()
    for var in p:
        lbl = entry.labels(typ)[var]
        if lbl.human:
            preds.append((var, p[var], lbl.label, counts[var]))

    return (preds, entry.meta)


def main(args):
    corpus = Corpus(args.input, full_strip=args.strip)

    with multiprocessing.Pool(processes=args.nproc, initializer=init, initargs=(args,)) as pool:
        preds = list(tqdm(pool.imap_unordered(predict_one, corpus), desc='Predicting'))

    out = []
    for inner, meta in preds:
        for var, pred, label, count in inner:
            item = {
                'var': var,
                'pred': pred,
                'label': label,
                'count': count,
            }
            item.update(meta)
            out.append(item)

    df = pd.DataFrame(out)
    df.to_csv(args.output, index=False)

    print('Wrote %d predictions to %s' % (len(preds), args.output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Path to input.jsonl (STRIDE format)')
    parser.add_argument('vocab', help='Path to label.vocab')
    parser.add_argument('output', help='Path to output.csv')
    parser.add_argument('--dbs', '-d', nargs='+', help='Path to ngram.db')
    parser.add_argument('--type', '-t', choices=['name', 'type'], default='name', help='Label type')
    parser.add_argument('--nproc', '-p', type=int, default=None, help='Number of processes')
    parser.add_argument('--flanking', '-f', action='store_true', help='Use flanking ngrams', default=False)
    parser.add_argument('--strip', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
