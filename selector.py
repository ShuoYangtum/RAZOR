import numpy as np
from prompt import *

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
    angle_rads[0::2] = np.sin(angle_rads[0::2])
    angle_rads[1::2] = np.cos(angle_rads[1::2])
    return angle_rads

def get_angle(pos, i):
    return pos / np.power(10000, (2 * (i // 2)) / 8)