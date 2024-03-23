
import argparse

from ..corpus import Corpus
from ..vocab import Vocab
from ..ngram import build_ngram_db_multi


def main(args):
    corpus = Corpus(args.input, full_strip=args.strip)
    vocab = Vocab.load(args.vocab)
    db = build_ngram_db_multi(corpus, vocab, args.type, args.size, args.topk, args.flanking)
    db.save(args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Path to input.jsonl (STRIDE format)')
    parser.add_argument('vocab', help='Path to input.vocab')
    parser.add_argument('output', help='Path to output.db')
    parser.add_argument('--type', '-t', choices=['name', 'type'], default='name', help='Which type of db to build')
    parser.add_argument('--size', '-s', type=int, default=3, help='Ngram size')
    parser.add_argument('--topk', '-k', type=int, default=5, help='Number of top-k targets to store')
    parser.add_argument('--flanking', '-f', action='store_true', help='Use flanking ngrams', default=False)
    parser.add_argument('--strip', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
