
import json
from typing import List, Tuple, Set, Dict, Generator
import re

from .ngram import ngram_hash

GHIDRA_STACK = re.compile('[a-z]*Stack_[0-9]+')

GHIDRA_VAR = re.compile('[a-z]*Var[0-9]+')

GHIDRA_ADDR_STRING = re.compile('s_[a-zA-Z0-9_]+[a-fA-F0-9]{8}')
GHIDRA_ADDR_PTR = re.compile('PTR_[a-zA-Z0-9_]+[a-fA-F0-9]{8}')

HEXNUM = re.compile('0x[0-9a-fA-F]+')
DECNUM = re.compile('[0-9]+')

PREFIXES = [
    # IDA
    'sub_', 'loc_', 'unk_', 'off_', 'asc_', 'stru_', 'funcs_',
    'byte_', 'word_', 'dword_', 'qword_', 'xmmword_', 'ymmword_',
    'LABEL_',

    # Ghidra
    'FUN_', 'thunk_FUN_', 
    'LAB_', 'joined_r0x', # goto labels
    'DAT_', '_DAT_', 'code_r0x', 'uRam', # data and code labels 
    'switchD_', 'switchdataD_', 'caseD_', # switch labels
]

FULL_STRIP = '''
?
Number
String
L

__int8
__int16
__int32
__int64
LODWORD
const
_BYTE
_WORD
_DWORD
_QWORD
char
float
double
__fastcall
unsigned
void

break
if
else
while
int
void
goto

[
]
(
)
{
}
+
-
,
;
*
*=
<
>
=
<=
>=
==
!=
++
--
+=
-=
<<
>>
<<=
>>=
!
|
||
|=
&
&&
&=
/
/=
^
^=
%
%=
'''

FULL_STRIP = [x for x in FULL_STRIP.split() if x != '']

class Corpus(object):
    def __init__(self, path, full_strip=False):
        self.path = path
        self.full_strip = full_strip

    def __iter__(self):
        with open(self.path, 'r') as f:
            for line in f:
                yield Entry(json.loads(line), full_strip=self.full_strip)


class Entry(object):
    def __init__(self, raw, full_strip=False):
        self.raw = raw
        self._stripped_tokens = None
        self.full_strip = full_strip

    @property
    def tokens(self) -> List[str]:
        return self.raw['tokens']
    
    @property
    def stripped_tokens(self) -> List[str]:
        if self._stripped_tokens is not None:
            return self._stripped_tokens

        out = []
        for tok in self.tokens:
            norm = tok
            for prefix in PREFIXES:
                if tok.startswith(prefix):
                    norm = prefix + 'XXX'
                    break
            
            if tok.startswith('"') and tok.endswith('"'):
                norm = '<STRING>'

            if GHIDRA_STACK.fullmatch(tok):
                norm = '<ghidra_stack>'

            if GHIDRA_VAR.fullmatch(tok):
                norm = '<ghidra_var>'

            if GHIDRA_ADDR_STRING.fullmatch(tok):
                norm = tok[:-8]

            if GHIDRA_ADDR_PTR.fullmatch(tok):
                norm = tok[:-8]

            if HEXNUM.fullmatch(tok):
                val = int(tok, 16)
                if val >= 0x100:
                    hex_digits = len(hex(val)[2:])
                    norm = '<NUM_%d>' % hex_digits
                else:
                    norm = hex(val)

            if DECNUM.fullmatch(tok):
                val = int(tok)
                if val >= 0x100:
                    hex_digits = len(hex(val)[2:])
                    norm = '<NUM_%d>' % hex_digits
                else:
                    norm = hex(val)

            out.append(norm)

        if self.full_strip:
            out = [x if x in FULL_STRIP else '?' for x in out]

        self._stripped_tokens = out
        return out

    @property
    def meta(self) -> Dict:
        if 'meta' in self.raw:
            return self.raw['meta']
        return {}

    def labels(self, label):
        return Labels(self.raw['labels'][label])
    
    def all_vars(self) -> Set[str]:
        return set(x[2:-2] for x in self.tokens if x.startswith('@@') and x.endswith('@@'))
    
    def var_counts(self) -> Dict[str, int]:
        '''Returns a dictionary of variable -> number of human labels.'''
        counts = {}
        for tok in self.tokens:
            if tok.startswith('@@') and tok.endswith('@@'):
                var = tok[2:-2]
                if var not in counts:
                    counts[var] = 0
                counts[var] += 1
        return counts

    def ngram_span(self, index: int, size: int) -> List[str]:
        '''Returns the N-gram centered at the given index.'''
        padded = ['??'] * size + self.stripped_tokens + ['??'] * size
        return padded[index:index+size*2+1]
    
    def ngram_hash(self, index: int, size: int) -> bytes:
        '''Returns the hash of the N-gram centered at the given index.'''
        return ngram_hash(self.ngram_span(index, size))

    def iter_ngrams(self, size: int, flanking: bool = False) -> Generator[Tuple[bytes, List[str], any, str], None, None]:
        '''Iterates over all N-grams of a given size in the function. Returns a tuple of (hash, span, idx, variable).'''
        padded = ['??'] * size + self.stripped_tokens + ['??'] * size
        for i in range(len(self.tokens)):
            if self.tokens[i].startswith('@@') and self.tokens[i].endswith('@@'):
                var = self.tokens[i][2:-2]

                if not flanking:
                    # centered ngrams
                    span = padded[i:i+size*2+1]
                    h = ngram_hash(span)
                    yield (h, span, i, var)
                else:
                    # flanking ngrams
                    # . . [ . . . {C] . . . } . .

                    # center is at size + i

                    left_span = padded[i:i+size]
                    right_span = padded[i+size+1:i+size*2+1]

                    left_h = ngram_hash(left_span, b'left')
                    right_h = ngram_hash(right_span, b'right')

                    yield (left_h, left_span, (i, False), var)
                    yield (right_h, right_span, (i, True), var)


class Labels(object):
    def __init__(self, raw):
        self.raw = raw

    def __getitem__(self, label):
        return Label(self.raw[label])
    
    def all_human_labels(self) -> List[str]:
        return [v['label'] for _, v in self.raw.items() if v['human']]


class Label(object):
    def __init__(self, raw):
        self.raw = raw

    @property
    def human(self) -> bool:
        return self.raw['human']
    
    @property
    def label(self) -> str:
        return self.raw['label']
