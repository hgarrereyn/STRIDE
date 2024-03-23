
import argparse
import json
from pathlib import Path
import tarfile

from tqdm.auto import tqdm


def convert_one(info, is_test=False):
    
    source = info['source']
    target = info['target']

    name_info = {}
    type_info = {}

    for k in source:
        # original name (matches token)
        var = source[k]['n']

        target_name = target[k]['n']
        target_type = json.dumps(target[k]['t'])

        if target_name == '':
            target_name = '<EMPTY>'

        name_info[var] = {
            'label': target_name,
            'human': True,
        }

        type_info[var] = {
            'label': target_type,
            'human': True,
        }

    out = {
        'tokens': info['code_tokens'],
        'labels': {
            'name': name_info,
            'type': type_info,
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
    parser.add_argument('input', help='Path to DIRT input folder')
    parser.add_argument('output', help='Path to output folder')
    args = parser.parse_args()
    main(args)
