
from typing import List, Tuple

import h5py


class NGramDBMulti(object):
    def __init__(self, size: List[int], hsh, total, typ, counts):
        self.size = size
        self.hsh = hsh
        self.total = total
        self.typ = typ
        self.counts = counts

    def __repr__(self):
        return f'NGramDBMulti(size={self.size}, count={len(self.hsh)})'

    def save(self, fpath):
        with h5py.File(fpath, 'w') as f:
            f['size'] = self.size
            f['hsh'] = self.hsh
            f['total'] = self.total
            f['typ'] = self.typ
            f['counts'] = self.counts

    @staticmethod
    def load(fpath) -> 'NGramDBMulti':
        with h5py.File(fpath, 'r') as f:
            size = f['size'][()]
            hsh = f['hsh'][()]
            total = f['total'][()]
            typ = f['typ'][()]
            counts = f['counts'][()]

        return NGramDBMulti(size, hsh, total, typ, counts)

    def lookup(self, key) -> Tuple[int, List[Tuple[int, int]]]:
        a = 0
        b = len(self.hsh)

        while a < b:
            m = (a + b) // 2

            q = self.hsh[m]
            if q == key:
                total = self.total[m]
                typ = self.typ[m]
                counts = self.counts[m]
                return total, list(zip(typ, counts))
            elif q < key:
                a = m+1
            elif q > key:
                b = m

        return None
