
import argparse

from tqdm.auto import tqdm

from ..corpus import Corpus
from ..vocab import Vocab


def main(args):
    vocab = Vocab.build_vocab(Corpus(args.input), args.type)
    vocab.save(args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Path to input.jsonl (STRIDE format)')
    parser.add_argument('output', help='Path to output.vocab')
    parser.add_argument('--type', '-t', choices=['name', 'type'], default='name', help='Which type of vocab to build')
    args = parser.parse_args()
    main(args)
