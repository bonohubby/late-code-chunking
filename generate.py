import os
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def truncate(data, tokenizer, side, max_num_tokens):
    tokens = tokenizer.tokenize(data)
    num_tokens = len(tokens)
    if side == 'left':
        prompt_tokens = tokens[num_tokens - max_num_tokens:]
    elif side == 'right':
        prompt_tokens = tokens[:max_num_tokens]
    else:
        print('Invalid Side:', side)
        exit(1)

    return tokenizer.convert_tokens_to_string(prompt_tokens)


def prepare_prompt(dataset, tokenizer, prompt_budget, cross_file_budget, prompt_type):
    prompt = truncate(dataset['prompt'], tokenizer, 'left', prompt_budget)
    if prompt_type == "in-file":
        return prompt
    else:
        crossfile_context = truncate(dataset['crossfile_context']['text'], tokenizer, 'right', cross_file_budget)
        return f'{crossfile_context}\n{prompt}'


model = os.environ['MODEL']
tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

input = os.environ['INPUT']
output = os.environ['OUTPUT']

prompt_type = os.environ['PROMPT_TYPE']

temperature = 0.0
top_p = 0.95
max_model_len = 8192
generation_max_tokens = 50
crossfile_max_tokens = 4096
buffer = 100

prompts = []
datasets = []

with open(input) as f:
    for line in f:
        dataset = json.loads(line.strip())
        datasets.append(dataset)

        prompt = prepare_prompt(
            dataset,
            tokenizer,
            max_model_len - generation_max_tokens - crossfile_max_tokens - buffer,
            crossfile_max_tokens,
            prompt_type
        )
        prompts.append(prompt)

sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=generation_max_tokens)
llm = LLM(model=model, max_model_len=max_model_len)

preds = llm.generate(prompts, sampling_params)
with open(output, 'w') as f:
    for pred, dataset in zip(preds, datasets):
        data = json.dumps({
            'left': dataset['prompt'].split('\n')[-1],
            'pred': pred.outputs[0].text,
            'gt': dataset['groundtruth'],
            'meta': dataset['metadata']
        })
        f.write(f'{data}\n')
