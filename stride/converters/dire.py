
import argparse
import json
from pathlib import Path
import tarfile

from tqdm.auto import tqdm


def _get_dire_varmap(node, vmap):
    if node['node_type'] == 'var':
        var_id = node['var_id']
        old_name = node['old_name']
        new_name = node['new_name']
        
        vmap[var_id] = (old_name, new_name)
    elif 'children' in node:
        for c in node['children']:
            _get_dire_varmap(c, vmap)
            

def get_dire_varmap(ast):
    vmap = {}
    _get_dire_varmap(ast, vmap)
    return vmap


def convert_one(info, is_test=False):
    vmap = get_dire_varmap(info['ast'])

    name_info = {}

    for k in vmap:
        # original name (matches token)
        var = vmap[k][0]
        target_name = vmap[k][1]

        if target_name == '':
            target_name = '<EMPTY>'

        name_info[var] = {
            'label': target_name,
            'human': True,
        }

    # find unnamed variables
    for tok in info['code_tokens']:
        if tok.startswith('@@') and tok.endswith('@@'):
            var = tok[2:-2]
            if var not in name_info:
                name_info[var] = {
                    'label': '<none>',
                    'human': False,
                }

    out = {
        'tokens': info['code_tokens'],
        'labels': {
            'name': name_info,
        },
    }

    if is_test:
        out['meta'] = {
            'fit': info['test_meta']['function_body_in_train']
        }

    return json.dumps(out)


def load_files(files, is_test=False):
    out = []

    for file in tqdm(files):
        t = tarfile.open(file)
        for member in tqdm(t.getmembers()):
            b = member.name.split('_')[0]

            data = t.extractfile(member).read()
            data = data.decode('ascii').strip().split('\n')
            for line in data:
                info = json.loads(line)

                entry = convert_one(info, is_test=is_test)
                out.append(entry)

    return out


def main(args):
    train = list(Path(args.input).glob('train-shard-*.tar'))
    test = [Path(args.input) / 'test.tar']
    
    train = load_files(train)
    test = load_files(test, is_test=True)

    with open(Path(args.output) / 'converted_train.jsonl', 'w') as f:
        f.write('\n'.join(train))

    with open(Path(args.output) / 'converted_test.jsonl', 'w') as f:
        f.write('\n'.join(test))

    print('Wrote %d train and %d test entries to %s' % (len(train), len(test), args.output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Path to DIRE input folder')
    parser.add_argument('output', help='Path to output folder')
    args = parser.parse_args()
    main(args)
