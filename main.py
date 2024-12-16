from transformers import AutoTokenizer, AutoModel
import torch
import math
import numpy as np
import json
from tqdm import tqdm
import re
from openai import OpenAI
import random
from sentence_transformers import CrossEncoder
from scipy.spatial.distance import cosine
import torch.nn.functional as F
from prompt import *
from selector import *
from calculator import *
import argparse

def main():
    parser = argparse.ArgumentParser(description="The hyper-parameters for RAZOR.")
    parser.add_argument('--data_pth', type=str, help="Your input json file")
    parser.add_argument('--output_pth', type=str, help="Your output json path")
    parser.add_argument('--model_name', type=str, help="The name of model", default="bert-base-uncased")
    parser.add_argument('--api_key', type=str, help="Your api key for OpenAI")
    parser.add_argument('--task', type=str, help="Fever or SA")
    
    args = parser.parse_args()
    data_pth=args.data_pth
    output_pth=args.output_pth
    model_name=args.model_name
    api_key=args.api_key
    TASK=args.task
    client = OpenAI(api_key=api_key)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab={key:0 for key in tokenizer.vocab}

    # for training hyper-parameters adjusting, please check them here
    BEAM_WIDTH=2
    MODEL='gpt-3.5-turbo'
    K=50 
    N=1 
    method='cos'    
    position=True
    p_d=8
    EMB=False
    epoch_num=1
    reverse=False

    if TASK=='SA':
        with open(data_pth) as f:
            data=json.load(f)
    elif TASK=='fever':
        with open(data_pth) as f:
            data=[json.loads(line) for line in f.readlines()]

    if TASK=='SA':
        pos_label='positive'
        neg_label='negative'
    elif TASK=='fever':
        pos_label='SUPPORTS'
        neg_label='REFUTES'
    
    pos_sents=[] 
    neg_sents=[] 
    if TASK=='SA':
        for sample in data:
            if sample['polarity'] == pos_label:
                pos_sents.append(sample['sentence'])
            elif sample['polarity'] == neg_label:
                neg_sents.append(sample['sentence'])
            else:
                print('Warning: The following sample dose not hold a standard label:', sample['polarity'])
    elif TASK=='fever':
        nli_model=CrossEncoder('cross-encoder/nli-roberta-base')
        for sample in data:
            if sample['lable'] == pos_label:
                claim=sample['claim']
                evidence=sample['evidence']
                pos_sents.append(f'Evidence:"{" ".join(evidence)}"\nClaim:"{claim}"')
            elif sample['lable'] == neg_label:
                claim=sample['claim']
                evidence=sample['evidence']
                neg_sents.append(f'Evidence:"{" ".join(evidence)}"\nClaim:"{claim}"')

    pos_sents=pos_sents
    neg_sents=neg_sents
    neg_text_dis=transfer_text_to_pro_distribution(neg_sents, tokenizer, vocab, n=N, \
                                                   positional_encoding=position, d_model=p_d, embedding=EMB, model_name=model_name) 
    pos_text_dis=transfer_text_to_pro_distribution(pos_sents, tokenizer, vocab, n=N, \
                                                   positional_encoding=position, d_model=p_d, embedding=EMB, model_name=model_name) 
    for e in range(epoch_num):
        print('Epoch: ', e)
        pos_sents=revise_sents(pos_sents, neg_text_dis, tokenizer, vocab, k=K, label=pos_label,\
                               task=TASK, model=MODEL, N=N, beam_width=BEAM_WIDTH, reverse=reverse,\
                               method=method, extra_model=nli_model, positional_encoding=position,\
                               d_model=p_d, embedding=EMB, model_name='bert-base-uncased')
        pos_text_dis=transfer_text_to_pro_distribution(pos_sents, tokenizer, vocab, n=N,\
                                positional_encoding=position, d_model=p_d, embedding=EMB, model_name='bert-base-uncased')
        neg_sents=revise_sents(neg_sents, pos_text_dis, tokenizer, vocab, k=K, label=neg_label,\
                               task=TASK, model=MODEL, N=N, beam_width=BEAM_WIDTH, reverse=reverse, \
                               method=method, extra_model=nli_model, positional_encoding=position,\
                               d_model=p_d, embedding=EMB, model_name='bert-base-uncased')
        neg_text_dis=transfer_text_to_pro_distribution(neg_sents, tokenizer, vocab, n=N,\
                                positional_encoding=position, d_model=p_d, embedding=EMB, model_name='bert-base-uncased')
    with open(output_pth, 'w') as f:
        revised_data=list()
        if TASK=='SA':
            for sent in pos_sents:
                revised_data.append({'sentence':sent, 'polarity':'positive'})
            for sent in neg_sents:
                revised_data.append({'sentence':sent, 'polarity':'negative'})
            random.shuffle(revised_data)
            json.dump(revised_data, f)
        elif TASK=='fever':
            for sent in pos_sents:
                revised_data.append({'evidence':[sent.split('\nClaim:')[0][10:-1]],\
                                     'claim':sent.split('\nClaim:')[-1][1:-1], 'lable':'SUPPORTS'})
            for sent in neg_sents:
                revised_data.append({'evidence':[sent.split('\nClaim:')[0][10:-1]],\
                                     'claim':sent.split('\nClaim:')[-1][1:-1], 'lable':'REFUTES'})
            random.shuffle(revised_data)
            for d in revised_data:
                f.write(json.dumps(d)+'\n')

if __name__=='__main__':
    main()