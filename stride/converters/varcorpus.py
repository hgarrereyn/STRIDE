
import argparse
import json
import multiprocessing

from tqdm.auto import tqdm

from ..lexer import Lexer


def varcorpus_tokenize(raw):
    tokens = []
    variables = {}

    for tok in Lexer(raw).get_tokens():
        if tok[1].startswith('@@') and tok[1].endswith('@@'):
            var, name = tok[1][2:-2].split('@@')
            variables[var] = name
            tokens.append('@@' + var + '@@')
        else:
            tokens.append(tok[1])

    return tokens, variables


def convert_one(line):
    before = json.loads(line)
    tokens, variables = varcorpus_tokenize(before['func'])

    # "vars_map" maps from the original variable name to the normalized variable name which is used for training/eval.
    vars_map = {z[0]: z[1] for z in before['vars_map']}

    name_info = {}
    for var, name in variables.items():
        name_info[var] = {
            # Name is only sometimes in vars_map
            'label': vars_map[name] if name in vars_map else name,
            'human': before['type_stripped_vars'][name] == 'dwarf',
        }

    info = {
        'tokens': tokens,
        'labels': {
            'name': name_info
        },
        'meta': {
            'id': before['id'],
            'func_name': before['func_name'],
        }
    }

    return json.dumps(info)


def main(args):
    lines = open(args.input).readlines()

    with multiprocessing.Pool() as pool:
        converted = list(tqdm(pool.imap_unordered(convert_one, lines), total=len(lines)))

    with open(args.output, 'w') as f:
        f.write('\n'.join(converted))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()
    main(args)
