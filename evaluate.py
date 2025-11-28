import os
import re
import json
from thefuzz import fuzz


lang = os.environ['LANG']

output = os.environ['OUTPUT']
root, ext = os.path.splitext(output)
eval_output = f'{root}_eval{ext}'


def _remove_comments(code):
    code = re.sub(r'#.*', '', code)
    code = re.sub(r'//.*', '', code)
    return code


def pre_process(code):
    pre_processed_code = _remove_comments(code)
    return [l.strip() for l in pre_processed_code.split("\n") if l.strip()]


def exact_match(pred, gt, lang):
    if len(pred) == 0:
        return False

    if lang == 'python':
        return pred[0] == gt[0]

    if lang in ['java', 'csharp', 'typescript']:
        pred_concat = '\n'.join(pred)
        gt_concat = '\n'.join(gt)
        for idx in range(len(pred_concat)):
            if pred_concat[idx] in [';', '{', '}']:
                break
        return pred_concat[:idx + 1] == gt_concat
    
    print('EM Error')
    exit(1)


def edit_distance(pred, gt, lang):
    if len(pred) == 0:
        return 0

    if lang == 'python':
        return fuzz.ratio(pred[0], gt[0])

    if lang in ['java', 'csharp', 'typescript']:
        pred_concat = '\n'.join(pred)
        gt_concat = '\n'.join(gt)
        for idx in range(len(pred_concat)):
            if pred_concat[idx] in [';', '{', '}']:
                break
        return fuzz.ratio(pred_concat[:idx + 1], gt_concat)

    print('ES Error')
    exit(1)


em_count = 0
es_sum = 0
total = 0
eval_results = []
with open(output) as f, open(eval_output, 'w') as f_out:
    for line in f:
        total += 1
        data = json.loads(line.strip())
        pred = pre_process(data['pred'])
        gt = pre_process(data['gt'])

        em = 1 if exact_match(pred, gt, lang) else 0
        em_count += em

        es = edit_distance(pred, gt, lang)
        es_sum += es
        data['eval'] = {
            'em': em,
            'es': es
        }
        f_out.write(f'{json.dumps(data)}\n')

print(f'Total:{total}, EM Count:{em_count}')

final_em = round(em_count / total * 100, 2)
final_es = round(es_sum / total, 2)

print('EM:', final_em)
print('ES:', final_es)
