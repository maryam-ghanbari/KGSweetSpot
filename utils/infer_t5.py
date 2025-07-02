import argparse
import json
import os
import random
import time
from tqdm import tqdm
import numpy as np
import torch
import transformers
import math

# Function to set seeds for reproducibility
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

import torch
import torch.nn.utils.rnn as rnn_utils

global empty_counter
empty_counter = 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def score_for_input_batch(args, tokenizer, model, query, cands, knowledge_batches):
    """
    Batch processes multiple knowledge contexts.
    `knowledge_batches` is a list of strings; each string is a concatenation of one or more knowledge statements.
    """
    model.eval()
    sources = []
    for kb in knowledge_batches:
        if args.task == 'csqa':
            if 'flan-t5' in args.model_type:
                source = f"Context: {kb} \n Question: {query} \n Choices: " + \
                         ' '.join([f'({chr(ord("a") + i)}) {cand}' for i, cand in enumerate(cands)]) + \
                         "\n Provide only the correct answer word."
                targets = cands
            elif 'unifiedqa-t5' in args.model_type or args.model_ckpt is not None:
                source = f"{kb} \\n {query} \\n " + \
                         ' '.join([f'({chr(ord("a") + i)}) {cand}' for i, cand in enumerate(cands)])
                targets = cands
            elif 't5' in args.model_type:
                source = query
                if kb is not None:
                    source = f"{kb} {source}"
                targets = [f'<extra_id_0> {cand} <extra_id_1>' for cand in cands]
            else:
                raise Exception(f'score_for_input_batch not implemented for {args.task} {args.model_type}!')
        else:
            source = query
            targets = cands
        sources.append(source)

    # Batch encode the sources without padding/truncation.
    encoded_sources = tokenizer(sources, return_tensors=None, padding=False, truncation=False)["input_ids"]
    # Convert the list of token lists to a list of tensors without an explicit for loop.
    tokenized_sources = list(map(torch.tensor, encoded_sources))
    # Pad them to the same length.
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(tokenized_sources, batch_first=True,
                                                       padding_value=tokenizer.pad_token_id)
    # Create an attention mask: 1 for real tokens, 0 for padding.
    attention_mask = (padded_input_ids != tokenizer.pad_token_id).long()

    batch_size = padded_input_ids.shape[0]
    padded_input_ids = padded_input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # For each source, we need to evaluate all candidate answers.
    num_cands = len(cands)
    input_ids = padded_input_ids.repeat_interleave(num_cands, dim=0)
    attn_mask = attention_mask.repeat_interleave(num_cands, dim=0)
    
    # Prepare flattened targets for each candidate for each source.
    if args.task == 'csqa' and 't5' in args.model_type and not (
         'flan-t5' in args.model_type or 'unifiedqa-t5' in args.model_type):
        flattened_targets = [f'<extra_id_0> {cand} <extra_id_1>' for _ in range(batch_size) for cand in cands]
    else:
        flattened_targets = [cand for _ in range(batch_size) for cand in cands]
    
    # Batch encode targets without padding/truncation.
    encoded_targets = tokenizer(flattened_targets, return_tensors=None, padding=False, truncation=False)["input_ids"]
    tokenized_targets = list(map(torch.tensor, encoded_targets))
    padded_labels = torch.nn.utils.rnn.pad_sequence(tokenized_targets, batch_first=True,
                                                    padding_value=tokenizer.pad_token_id)
    labels = padded_labels.to(device)
    chunk_len = 300
    chunk_num = math.ceil(input_ids.shape[0]/chunk_len)
    all_logits = []
    #breakpoint()
    # Forward pass with the model.
    if chunk_num == 0:
        loss_out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        logits = loss_out[1]
    else:
        for i in range(chunk_num):
            if i == chunk_num - 1:
              loss_out = model(input_ids=input_ids[chunk_len*i:], attention_mask=attn_mask[chunk_len*i:], labels=labels[chunk_len*i:])
            else:
                loss_out = model(input_ids=input_ids[chunk_len*i:chunk_len*(i+1)], 
                                attention_mask=attn_mask[chunk_len*i:chunk_len*(i+1)], labels=labels[chunk_len*i:chunk_len*(i+1)])
            logits = loss_out[1]
            all_logits.append(logits)
        #breakpoint()
        all_logits = torch.cat(all_logits, dim=0)
        logits = all_logits
    
    # Compute the loss manually using a mask to ignore pad tokens.
    loss_function = torch.nn.CrossEntropyLoss(reduction='none')
    seq_length = labels.shape[-1]
    mask = (labels != tokenizer.pad_token_id).int().to(labels.device)
    loss_val = loss_function(logits.view(-1, logits.shape[-1]), labels.view(-1))
    loss_val = loss_val.view(batch_size * num_cands, seq_length)
    # Sum the loss over the sequence length, masking out pad tokens.
    loss_val = (loss_val * mask).sum(-1)
    
    # Reshape loss into (num_batches, num_candidates)
    loss_val = loss_val.view(len(knowledge_batches), num_cands)
    # Use negative loss as the score.
    selected_logits = -loss_val
    probs = torch.softmax(selected_logits, dim=-1)
    
    return selected_logits, probs


def score_for_query(args, tokenizer, model, query, knowledges, cands):
    global empty_counter
    # If knowledges is a list, process them in batches
    if isinstance(knowledges, list):
        n = len(knowledges)
        # If no knowledge is provided, process without context.
        if n == 0:
            print("No knowledge provided; processing without context.")
            empty_counter += 1
            knowledge_batches = []
            score , probs = score_for_input_batch(args, tokenizer, model, query, cands, [query])
            # Here we call score_for_input_batch with a dummy context (e.g., just the query)
            return score , probs , knowledge_batches
            
        h, v = args.h, args.v
        if h == -1 and v == -1:
            raise Exception('h and v cannot be both -1!')
        if h * v > n:
            raise Exception('h*v must be no larger than the number of knowledges!')
        if h == -1:
            h = n // v
        if v == -1:
            v = n // h
        # Batch the knowledge statements into groups of size h
        knowledge_batches = ['\n'.join(knowledges[i:i+h]) for i in range(0, len(knowledges), h)]
                    
        scores, probs = score_for_input_batch(args, tokenizer, model, query, cands, knowledge_batches)
        return scores, probs, knowledge_batches
    # In case knowledges is not a list, you can add handling here if needed.

def checker(args, answer, pred):
    return 1 if answer == pred else 0

def process_item(args, tokenizer, model, item):
    query = item['query'] if 'query' in item else item['question']
    if 'cands' in item:
        cands = item['cands']
    elif args.task == 'csqa2':
        cands = ['yes', 'no']
    else:
        raise Exception(f'process_item() not implemented for {args.task}!')
    
    knowledges = item['knowledges'] if 'knowledges' in item else []
    random.shuffle(knowledges)
    scores_, probs_, knowledge_batches = score_for_query(args, tokenizer, model, query, knowledges, cands)

    item['knowledges'] = knowledge_batches if knowledge_batches else []
    # Aggregate scores across batches (e.g., by taking the mean)
    scores = torch.mean(scores_, dim=0)
    probs = torch.mean(probs_, dim=0)


    if args.aggfunc == 'best_score':
        p = scores.argmax().item()
    elif args.aggfunc == 'best_prob':
        p = probs.argmax().item()
    elif args.aggfunc == 'poe':
        probs_prod = torch.prod(torch.stack(probs_) if isinstance(probs_, list) else probs_, dim=0)
        p = probs_prod.argmax().item()
    elif args.aggfunc == 'moe':
        probs_sum = torch.sum(torch.stack(probs_) if isinstance(probs_, list) else probs_, dim=0)
        p = probs_sum.argmax().item()
    pred = cands[p]

    item['scores_'] = scores_.tolist()
    item['probs_'] = probs_.tolist()
    item['scores'] = scores.tolist()
    item['probs'] = probs.tolist()
    item['pred'] = pred

    if 'answer' in item:
        answer = item['answer']
        ok = checker(args, answer, pred)
        item['ok'] = ok

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['csqa', 'csqa2', 'qasc'])
    parser.add_argument('--model-type', type=str, required=True, choices=[
        't5-small', 't5-base', 't5-large', 't5-3b', 't5-11b',
        'allenai/unifiedqa-t5-large', 'allenai/unifiedqa-t5-3b', 'allenai/unifiedqa-t5-11b',
        'google/flan-t5-xl', 'google/flan-t5-large', 'google/flan-t5-small'
    ])
    parser.add_argument('--model-ckpt', type=str, default=None)
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--average-loss', action='store_true')
    parser.add_argument('--h', type=int, default=1)
    parser.add_argument('--v', type=int, default=-1)
    parser.add_argument('--aggfunc', type=str, default='best_score', choices=['best_score', 'best_prob', 'poe', 'moe'])
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--n', type=int, default=None)
    parser.add_argument('--entire-knowledge', action='store_true', help='Use the entire knowledge set as a single context for evaluation.')
    args = parser.parse_args()
    args.output_path = f'/content/qagnn-main/Inference_results/data/{args.task}/inference/inference_{"" if args.model_ckpt is None else "ft"}{args.model_type.split("/")[-1]}.{args.input_path.split("/")[-1]}'

    tokenizer = transformers.T5Tokenizer.from_pretrained(args.model_type)
    model = transformers.T5ForConditionalGeneration.from_pretrained(
        args.model_ckpt if args.model_ckpt is not None else args.model_type)
    model = model.to(device)
    model.eval()

    if args.interactive:
        while True:
            example = input(f'Enter a {args.task} example: ')
            if args.task == 'csqa':
                splits = example.split(' -- ')
                query, cands = splits[0], splits[1:]
                item = {'query': query, 'cands': cands}
                process_item(args, tokenizer, model, item)
                print(item['pred'], item['probs'])
            elif args.task == 'csqa2':
                item = {'query': example}
                process_item(args, tokenizer, model, item)
                print(item['pred'], item['probs'])
            else:
                raise Exception(f'Interactive mode not implemented for {args.task}')

    with open(args.input_path) as f:
        ds = json.load(f)
        if args.n is not None:
            ds = ds[:args.n]

    pbar = tqdm(ds)
    num, den = 0, 0
    for idx, item in enumerate(pbar, start=1):
        item['id'] = str(idx)
        process_item(args, tokenizer, model, item)
        if 'ok' in item:
            num += item['ok']
            den += 1
            pbar.set_postfix({'acc': num / den})
    print('Total number of questions with empty knowledge: ', empty_counter)

    with open(args.output_path, 'w') as f:
        json.dump(ds, f, indent=4)

if __name__ == '__main__':
    start_time = time.time()
    with torch.no_grad():
        main()
    end_time = time.time()
    print("Overall time: ", end_time - start_time)
