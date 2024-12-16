RAZOR: Sharpening Knowledge by Cutting Bias with Unsupervised Text Rewriting
====
This is the code for **RAZOR: Sharpening Knowledge by Cutting Bias with Unsupervised Text Rewriting**. <br>

We mitigate the spurious correlations between tokens and labels by rewriting the dataset using large language models, please check our paper [here](https://arxiv.org/abs/2412.07675).<br>

![Image text](https://github.com/ShuoYangtum/images/blob/main/RAZOR.PNG)

Data sets
----
All of the datasets we used are open-soursed.<br>
Fever dataset: [https://fever.ai/dataset/adversarial.html](https://fever.ai/dataset/adversarial.html)<br>
MNLI dataset: [https://paperswithcode.com/dataset/multinli](https://paperswithcode.com/dataset/multinli)<br>
SNLI dataset: [https://paperswithcode.com/dataset/snli](https://paperswithcode.com/dataset/snli)<br>

Dependencies
----
Before running our code, please ensure that the following dependencies are met.<br> 

torch==2.3.0<br> 

tokenizers==0.19.1<br> 

tqdm==4.66.4<br> 

transformers==4.40.1<br> 

spacy==3.7.4<br> 

shap==0.46.0<br> 

sentence-transformers==3.0.1<br> 

pandas==2.2.2<br> 

openai==1.27.0<br> 

numpy==1.26.4<br> 

Running
----
To run our program, you can simply execute the main.py file located in the root directory.<br> 

The directory of the files and some commonly used hyperparameters can be passed via the command line.<br> 

Please note that hyperparameters used during training need to be manually adjusted by modifying the relevant sections of the main.py code.<br> 

Cited
----
If you are interested in our work or want to use our code, please use the following citation information.<br> 

@article{yang2024razor,<br> 
  title={RAZOR: Sharpening Knowledge by Cutting Bias with Unsupervised Text Rewriting},<br> 
  author={Yang, Shuo and Prenkaj, Bardh and Kasneci, Gjergji},<br> 
  journal={arXiv preprint arXiv:2412.07675},<br> 
  year={2024}<br> 
}


