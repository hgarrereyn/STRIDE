
import hashlib
import multiprocessing
from typing import List

from tqdm.auto import tqdm
import numpy as np

from .db import NGramDBMulti
from .vocab import Vocab


def ngram_hash(tokens, discriminator: bytes = b''):
    name_idx = 0
    normalized_names = {}

    out = []
    for tok in tokens:
        if tok.startswith('@@') and tok.endswith('@@'):
            var = tok[2:-2]

            if var not in normalized_names:
                normalized_names[var] = '@@var_%d@@' % name_idx
                name_idx += 1

            out.append(normalized_names[var])
        else:
            out.append(tok)

    raw = b'\xff'.join(tok.encode('utf-8') for tok in out) + discriminator
    return hashlib.sha256(raw).digest()[:12]


def process_one(entry: 'Entry', label: str, size: int, flanking: bool = False):
    labels = entry.labels(label)

    # hash -> {var -> count}
    hmap = {}

    for hsh, _span, _idx, var in entry.iter_ngrams(size, flanking=flanking):
        if not labels[var].human:
            # Skip non-human labels
            continue

        target = labels[var].label

        if hsh not in hmap:
            hmap[hsh] = {}
        if target not in hmap[hsh]:
            hmap[hsh][target] = 0
        hmap[hsh][target] += 1

    return hmap


class Processor:
    def __init__(self, label, size, flanking):
        self.label = label
        self.size = size
        self.flanking = flanking

    def __call__(self, entry: 'Entry'):
        return process_one(entry, self.label, self.size, self.flanking)


def build_ngram_db_multi(corpus: 'Corpus', vocab: Vocab, label: str, size: int, topk: int, flanking: bool) -> NGramDBMulti:
    proc = Processor(label, size, flanking)
    with multiprocessing.Pool() as pool:
        hmaps = list(tqdm(pool.imap_unordered(proc, corpus)))

    hmap = {}
    for sub in tqdm(hmaps, desc='Merging'):
        for h in sub:
            if not h in hmap:
                hmap[h] = sub[h]
            else:
                for target in sub[h]:
                    if target not in hmap[h]:
                        hmap[h][target] = 0
                    hmap[h][target] += sub[h][target]

    # (hash, total, [(t1, c2), (t2, c2), ..., (tk, ck)])
    entries = []

    for h in tqdm(hmap, desc='Max'):
        top = sorted(hmap[h].items(), key=lambda x: x[1], reverse=True)[:topk]
        top = [(vocab.lookup(x[0]), x[1]) for x in top if vocab.lookup(x[0]) is not None]

        # If there are fewer than topk unique targets, pad with nulls
        while len(top) < topk:
            top.append((0, 0))

        total = sum(x[1] for x in hmap[h].items())

        entries.append((h, total, top))
    

    print('Sorting...')
    entries.sort(key=lambda x: x[0])

    hsh = np.array([x[0] for x in entries], dtype='|S12')
    total = np.array([x[1] for x in entries], dtype=np.uint32)
    typ = np.array([[h[0] for h in x[2]] for x in entries], dtype=np.uint32)
    counts = np.array([[h[1] for h in x[2]] for x in entries], dtype=np.uint32)

    db = NGramDBMulti(size, hsh, total, typ, counts)
    print(db)
    return db
