
from typing import List

from tqdm.auto import tqdm

if __package__ is None or __package__ == '':
    from corpus import Corpus


class Vocab(object):
    def __init__(self, entries: List[str], counts: List[int]):
        self.entries = entries
        self.counts = counts
        self.map = {e: i for i, e in enumerate(entries)}

    def __repr__(self):
        return f'Vocab(count={len(self.entries)})'
    
    def save(self, fpath):
        with open(fpath, 'w') as f:
            for i in range(len(self.entries)):
                f.write(f'{self.entries[i]}\t{self.counts[i]}\n')

    @staticmethod
    def load(fpath) -> 'Vocab':
        entries = []
        counts = []
        with open(fpath, 'r') as f:
            for line in f:
                entry, count = line.strip().split('\t')
                entries.append(entry)
                counts.append(int(count))
        return Vocab(entries, counts)

    def lookup(self, key):
        return self.map.get(key, None)

    def reverse(self, id: int):
        return self.entries[id]

    def count_by_id(self, id: int):
        return self.counts[id]

    @staticmethod
    def build_vocab(corpus: 'Corpus', typ: str) -> 'Vocab':
        all_counts = {}

        for func in tqdm(corpus, desc='Building vocab'):
            counts = func.var_counts()
            labels = func.labels(typ)

            for var in counts:
                if not labels[var].human:
                    continue
                lbl = labels[var].label
                if lbl not in all_counts:
                    all_counts[lbl] = 0
                all_counts[lbl] += counts[var]
        
        pairs = [(k, v) for k, v in all_counts.items()]
        pairs.sort(key=lambda x: x[1], reverse=True)

        entries = []
        counts = []
        for k, v in pairs:
            entries.append(k)
            counts.append(v)

        return Vocab(entries, counts)
