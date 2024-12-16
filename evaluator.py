import os
import random
from sklearn.metrics import f1_score
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, load_metric
import json
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType

def preprocess_function(examples, use_evidence=True): 
    if use_evidence:
        return tokenizer(examples['claim'], examples['evidence'], truncation=True, padding='max_length')
    else:
        return tokenizer(examples['claim'], truncation=True, padding='max_length')

def acc(p):
    predictions = torch.tensor(p.predictions)
    references = torch.tensor(p.label_ids)
    return metric.compute(predictions=torch.argmax(predictions, axis=1), references=references)
def f1(p):
    predictions = torch.tensor(p.predictions)
    references = torch.tensor(p.label_ids)
    preds = torch.argmax(predictions, axis=1)
    f1 = f1_score(references, preds, average='binary')
    return {'f1': f1}
def compute_metrics(p):
    acc_result = acc(p)
    f1_result = f1(p)
    return {**acc_result, **f1_result}

def main():
    model_name='bert-base-uncased'
    lora=False
    
    with open('train.json') as f: 
        data=[json.loads(line) for line in f.readlines()]
    with open('dev.json') as f: 
        test_data=[json.loads(line) for line in f.readlines()]
    
    tmp_data=[]
    for d in data:
        if d['lable'] =='SUPPORTS':
            tmp_data.append({'claim':d['claim'], 'evidence':' '.join(d['evidence']), 'label':0})
        elif d['lable'] =='REFUTES':
            tmp_data.append({'claim':d['claim'], 'evidence':' '.join(d['evidence']), 'label':1})
    
    data=tmp_data
    data_dict = {
        "claim": [item["claim"] for item in data],
        "evidence": [item["evidence"] for item in data],
        "label": [item["label"] for item in data],
    }
    
    tmp_data=[]
    for d in test_data:
        if d['lable'] =='SUPPORTS':
            tmp_data.append({'claim':d['claim'], 'evidence':' '.join(d['evidence']), 'label':0})
        elif d['lable'] =='REFUTES':
            tmp_data.append({'claim':d['claim'], 'evidence':' '.join(d['evidence']), 'label':1})
            
    test_data=tmp_data
    test_data_dict = {
        "claim": [item["claim"] for item in test_data],
        "evidence": [item["evidence"] for item in test_data],
        "label": [item["label"] for item in test_data],
    }
    
    dataset = Dataset.from_dict(data_dict)
    test_dataset = Dataset.from_dict(test_data_dict)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    dataset = dataset.map(preprocess_function, batched=True)
    test_dataset = test_dataset.map(preprocess_function, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    if lora:
        peft_config = LoraConfig(
            r=16, 
            lora_alpha=16, 
            target_modules=["query", "value"],
            lora_dropout=0.1, 
            bias="none"
        )
        model = get_peft_model(model, peft_config)
        
    metric = load_metric("accuracy")
    
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=7,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=2000,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

if __name__=='__main__':
    main()