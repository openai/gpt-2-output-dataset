import argparse
import glob
import os
import torch
from transformers import (GPT2LMHeadModel, GPT2Tokenizer)
from tqdm import tqdm
import torch.nn as nn
from einops import rearrange
from parallelformers import parallelize

from baseline import _load_split


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, default='', help='input file or dir') # The torchtext way of loading data
    parser.add_argument('--output', '-o', type=str, default='', help='output file or dir')
    parser.add_argument('--min_len', type=int, default=2)
    parser.add_argument('--model', type=str, 
        choices=['gpt2'], 
        default='gpt2')
    return parser

def load_model(args):
    model_class = GPT2LMHeadModel
    tokenizer_class = GPT2Tokenizer
    pretrained_weights = 'gpt2'
    model = model_class.from_pretrained(pretrained_weights)
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    num_gpus = torch.cuda.device_count()
    parallelize(model, num_gpus=num_gpus, fp16=False, verbose='simple')
    return model, tokenizer

@torch.no_grad()
def process_single(input_file, output_file, model, tokenizer, args):
    device = model.device
    criterian = nn.NLLLoss(reduction='none')
    softmax = nn.Softmax(dim=1)
    log_softmax = nn.LogSoftmax(dim=1)

    # Read lines from input
    with open(input_file, 'r') as fr, open(output_file, 'w') as fw:
        for line in tqdm(fr.readlines()):
            line = line.strip()
            encoded_input = tokenizer(line, return_tensors='pt').to(device)
            input_ids = encoded_input['input_ids']
            if input_ids.shape[-1] < args.min_len: # Skip short ones whose length < args.min_len
                continue

            output = model(**encoded_input, labels= input_ids)
            logits = output.logits.to(device)
            target = encoded_input['input_ids'].to(device)

            logits = rearrange(logits, 'B L V -> B V L') # B=1
            shift_logits = logits[..., :, :-1] # Use the first L-1 tokens to predict the next
            shift_target = target[..., 1:]

            nll_loss = None
            if args.type == 'nll_loss':
                nll_loss = criterian(log_softmax(shift_logits), shift_target).squeeze()
                res = nll_loss.tolist()
            if not isinstance(res, list):
                res = [res]
            # Write to output
            try:
                res_str = ' '.join(f'{num:.4f}' for num in res)
            except Exception:
                print('line:', line)
                print('input_ids:', input_ids)
                print('logits.shape:', logits.shape)
                print('res:', res)
                raise
            else:
                fw.write(f'{res_str}\n')

def main():
    pass


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    model, tokenizer = load_model(args)
    if os.path.isfile(args.input):
        process_single(args.input, args.output, model, tokenizer, args)
    elif os.path.isdir(args.input):
        process_batch(model, tokenizer, args)