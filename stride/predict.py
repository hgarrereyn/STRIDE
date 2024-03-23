

from typing import List

import numpy as np

from .db import NGramDB, NGramDBMulti
from .corpus import Entry
from .vocab import Vocab


def predict_multi(entry: Entry, vocab: Vocab, dbs: List[NGramDBMulti], label: str, flanking: bool = False):
    '''
    Makes a prediction for all variables in a given function.

    Args:
        entry: The function to predict on (dict from STRIDE Corpus).
        vocab: The vocabulary to use.
        dbs: list of NGramDBs to use (sorted largest to smallest).
        label: The label to predict.
    '''
    locs = {}
    preds = {}

    for hsh, _span, idx, var in entry.iter_ngrams(1, flanking=flanking):
        if var not in locs:
            locs[var] = []
        locs[var].append(idx)
        preds[idx] = None

    for db in dbs:
        for hsh, span, idx, var in entry.iter_ngrams(db.size, flanking=flanking):
            if preds[idx] is not None:
                continue

            res = db.lookup(hsh)
            if res is None:
                continue

            total, targets = res

            entries = []
            for t, c in targets:
                if c == 0:
                    continue

                name = vocab.reverse(t)
                if name is None:
                    continue

                entries.append((name, c))

            preds[idx] = (db.size, total, entries)

    # Aggregate predictions by variable
    agg = {}
    for var in locs:
        agg[var] = {}
        for idx in locs[var]:
            if preds[idx] is None:
                continue

            size, total, entries = preds[idx]
            entry_total = sum(x[1] for x in entries)

            for name, count in entries:
                if name not in agg[var]:
                    agg[var][name] = 0
                
                # map ratio into [0.5, 1]
                score = (count / entry_total) * 0.5 + 0.5

                agg[var][name] += score

    # Find the best prediction for each variable
    for var in agg:
        best = None
        for name in agg[var]:
            if best is None or agg[var][name] > agg[var][best]:
                best = name
            elif agg[var][name] == agg[var][best]:
                # Tiebreack by vocab frequency
                if vocab.count_by_id(vocab.lookup(name)) > vocab.count_by_id(vocab.lookup(best)):
                    best = name

        agg[var] = best

    return agg


def predict_detailed(entry: Entry, vocab: Vocab, dbs: List[NGramDBMulti], label: str, flanking: bool = False):
    '''
    Makes a prediction for all variables in a given function.

    Args:
        entry: The function to predict on (dict from STRIDE Corpus).
        vocab: The vocabulary to use.
        dbs: list of NGramDBs to use (sorted largest to smallest).
        label: The label to predict.
    '''

    # [var] -> [idx, ...]
    locs = {}

    # [idx][db_size] -> [[name, count], ...]
    preds = {}

    for hsh, _span, idx, var in entry.iter_ngrams(1, flanking=flanking):
        if var not in locs:
            locs[var] = []
        locs[var].append(idx)
        preds[idx] = {}

    for db in dbs:
        for hsh, span, idx, var in entry.iter_ngrams(db.size, flanking=flanking):
            res = db.lookup(hsh)
            if res is None:
                preds[idx][int(db.size)] = []
                continue

            _total, targets = res

            entries = []
            for t, c in targets:
                if c == 0:
                    continue

                name = vocab.reverse(t)
                if name is None:
                    continue

                entries.append((name, int(c)))

            preds[idx][int(db.size)] = entries

    return preds, locs
