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

def remains_its_attribute(sent, label='positive', task='SA', model='gpt-3.5-turbo'):
    prompt=judge_prompt(given_sentence=sent, task=task)
    response = get_gpt_3_response(prompt=prompt, engine=model, temperature=0.1, stop=None)
    if label in response.lower() or label in response:
        return True
    else:
        return False

def revise_sent(sent, task='SA', model='gpt-3.5-turbo', label='positive'):
    prompt=revise_sent_prompt(sent, task=task, label=label)
    response = get_gpt_3_response(prompt=prompt, engine=model, temperature=0.8, stop=None)    
    pattern = r'"(.*?)"'
    try:
        matches = re.findall(pattern, response)
        return matches[-1]
    except:
        return response
        
def positional_encodings(pos, d_model=128):
    def get_angle(pos, i):
        return pos / np.power(10000, (2 * (i // 2)) / d_model)
    angle_rads = get_angle(pos, np.arange(d_model))
    # apply sin to even indices in the array; 2i
    angle_rads[0::2] = np.sin(angle_rads[0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[1::2] = np.cos(angle_rads[1::2])
    return angle_rads

def get_angle(pos, i):
    return pos / np.power(10000, (2 * (i // 2)) / 8)

def transfer_ngram_list_to_pro_distribution(n_grams, vocab, positional_encoding=False, d_model=128):
    num_tokens=len(n_grams)
    tmp_vocab=dict()
    for token in n_grams:
        if token not in tmp_vocab:
            tmp_vocab[token]=1
        else:
            tmp_vocab[token]+=1
    vocab=vocab.copy()
    for token in tmp_vocab:
        assert token in vocab, print('The following token cannot be found in vocab:', token)
        vocab[token]=tmp_vocab[token]/num_tokens
    if positional_encoding:
        for v in vocab:
            if vocab[v]==0:
                vocab[v]=np.zeros(d_model)
        for i in range(len(n_grams)):
            vocab[n_grams[i]]*=positional_encodings(i, d_model=d_model)
    return vocab
    
def transfer_sent_to_pro_distribution(sent, tokenizer, vocab, n=1, positional_encoding=False, d_model=128, embedding=False, model_name='bert-base-uncased'):
    device=torch.device('cuda:0')
    if embedding:
        model = AutoModel.from_pretrained(model_name).to(device)
        inputs = tokenizer(sent, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
        with torch.no_grad():
            outputs = model(**inputs.to(device))
        return outputs.last_hidden_state[:, 0, :].squeeze()
    else:
        vocab=vocab.copy()
        n_grams=n_gram_tokenize(sent, tokenizer, n=n)
        return transfer_ngram_list_to_pro_distribution(n_grams, vocab, positional_encoding=positional_encoding, d_model=d_model)


def dot_product(v1, v2):
    return sum(v1[key] * v2[key] for key in v1 if key in v2)

def vector_length(v):
    return math.sqrt(sum(v[key] ** 2 for key in v))

def cosine_similarity(v1, v2):
    dot = dot_product(v1, v2)
    length_v1 = vector_length(v1)
    length_v2 = vector_length(v2)
    if length_v1 == 0 or length_v2 == 0:
        return 0  
    return dot / (length_v1 * length_v2)

def n_gram_tokenize(sent, tokenizer, n=1):
    tokens=tokenizer.tokenize(sent)
    if n>len(tokens):
        print('Warning: N is larger than the sentence length.')
        return False
    elif n==1:
        return tokens
    else:
        n_grams=list()
        for i in range(len(tokens)-n+1):
            n_grams.append(' '.join(tokens[i:i+n]))
        return n_grams

def lists_to_matrix(lists):
    return np.array(lists)
    

def cosine_similarity_matrix(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)

def transfer_text_to_pro_distribution(sent_list, tokenizer, vocab, n=1, positional_encoding=False, d_model=128, embedding=False, model_name='bert-base-uncased'):
    if embedding:
        s=0
        for sent in tqdm(sent_list):
            s+=transfer_sent_to_pro_distribution(sent, tokenizer, vocab, embedding=embedding, model_name=model_name)
        return s/len(sent_list)
    vocab=vocab.copy()
    t_vocab=vocab.copy()
    count=len(sent_list)
    for sent in tqdm(sent_list):
        tmp_vocab=transfer_sent_to_pro_distribution(sent, tokenizer, t_vocab, n=n, positional_encoding=positional_encoding, d_model=d_model)
        for w in vocab:
            vocab[w]+=tmp_vocab[w]
    for w in vocab:
        vocab[w]/=count
    return vocab
    
def similarity_calculator(pos_sent_dis, neg_text_dis, method='cos', positional_encoding=False, embedding=False):
    if embedding:
        return float(F.cosine_similarity(pos_sent_dis.view(1, -1), neg_text_dis.view(1, -1))[0])
    if positional_encoding:
        pos_sent_dis=[pos_sent_dis[k] for k in pos_sent_dis]
        neg_text_dis=[neg_text_dis[k] for k in neg_text_dis]
        if method=='cos':
            similarities = []
            for row1, row2 in zip(pos_sent_dis, neg_text_dis):
                sim = cosine_similarity_matrix(row1, row2)
                similarities.append(sim)
            return np.mean(similarities)
        elif method=='kl':
            return kl_sim(pos_sent_dis, neg_text_dis)
        else:
            print('Error: You must select a correct similarity calculating method.')
    else:
        if method=='cos':
            return cosine_similarity(pos_sent_dis, neg_text_dis)
        elif method=='kl':
            return kl_sim(pos_sent_dis, neg_text_dis)
        else:
            print('Error: You must select a correct similarity calculating method.')

def revise_sents(pos_sents, neg_text_dis, tokenizer, vocab, k=10, label='positive', task='SA', \
                 model='gpt-3.5-turbo', N=1, beam_width=3, reverse=False, method='cos', \
                 extra_model=None, positional_encoding=False, d_model=128, embedding=False, model_name='bert-base-uncased'):

    scores=list()
    for pos_sent in tqdm(pos_sents):
        pos_sent_dis=transfer_sent_to_pro_distribution(pos_sent, tokenizer, vocab, n=N, positional_encoding=positional_encoding, \
                                                       d_model=d_model, embedding=embedding, model_name=model_name)
        scores.append((pos_sent, similarity_calculator(pos_sent_dis, neg_text_dis, method=method, positional_encoding=positional_encoding,\
                                                       embedding=embedding)))
    scores=sorted(scores, key=lambda x: x[1], reverse=reverse)
    sents=[sample[0] for sample in scores[:k]] 
    new_sents=[sample[0] for sample in scores[k:]] 
    for sent in sents:
        tmp_sents=[sent]
        for _ in range(beam_width):
            output=revise_sent(sent, task=task, model=model, label=label)
            if task=='SA':
                tmp_sents.append(output)
            elif task=='fever':
                evidence=sent.split('\n')[0]
                tmp_sents.append(evidence+f'\nClaim:"{output}"')
        revised_sents=select_the_best_revised_sent(sent, tmp_sents, neg_text_dis, tokenizer=tokenizer,vocab=vocab, \
                                                    label=label, task=task, model=model, N=N, beam_width=beam_width, \
                                                    reverse=not reverse, method=method, extra_model=extra_model, \
                                                    positional_encoding=positional_encoding, d_model=d_model,\
                                                    embedding=embedding, model_name=model_name)
        if revised_sents:
            new_sents.append(revised_sents[0]) 
        else:
            new_sents.append(sent)
    return new_sents

def select_the_best_revised_sent(ori_sent, sents, text_dis, tokenizer, vocab, label='positive', task='SA',\
                                 model='gpt-3.5-turbo', N=1, beam_width=3, reverse=True, method='cos', \
                                 extra_model=None, positional_encoding=False, d_model=128, embedding=False, model_name='bert-base-uncased'):
    scores=list()
    tmp_sents=list()
    for sent in sents:
        flag=True
        if sent in tmp_sents: 
            continue
        sent_dis=transfer_sent_to_pro_distribution(sent, tokenizer, vocab, n=N, positional_encoding=positional_encoding,\
                                                   d_model=d_model, embedding=embedding, model_name=model_name)
        judging_results=remains_its_attribute(sent, label=label, task=task, model=model)
        if judging_results:
            if extra_model:
                if label=='REFUTES':
                    s = extra_model.predict([(sent.split('\n')[0][len("Evidence:")+1:-1], sent.split('\n')[1][len("Claim:")+1:-1])])
                    if int(s.argmax(-1))!=0:
                        flag=False 
                
                elif label=='SUPPORTS':
                    s = extra_model.predict([(sent.split('\n')[0][len("Evidence:")+1:-1], sent.split('\n')[1][len("Claim:")+1:-1])])
                    if int(s.argmax(-1))!=1:
                        flag=False 
                
        else:
            flag=False
        if flag:
            scores.append((sent, similarity_calculator(sent_dis, text_dis, method=method,\
                                                        positional_encoding=positional_encoding,\
                                                      embedding=embedding)))
            tmp_sents.append(sent)
    scores=sorted(scores, key=lambda x: x[1], reverse=reverse)
    if scores:
        return [i[0] for i in scores[:beam_width]]
    else:
        print('Warning: No sentence is revised.')
        return None

def kl_sim(vocab1, vocab2):
    k=0
    for v in vocab1:
        p=vocab1[v]
        q=vocab2[v]
        if q==0:
            if p==0:
                continue
            else:
                q=1e-10
        if p==0:
            p=1e-10
        k+=p*math.log(p/q)
    return -k