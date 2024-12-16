from openai import OpenAI

def get_gpt_3_response(prompt, engine, temperature, stop):
    response = client.chat.completions.create(model=engine,
    messages=[
        {"role": "system", "content": "You are a text completion assistant."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=512,
    temperature=temperature,
    stop=stop,
    n=1)
    return response.choices[0].message.content

def judge_prompt(given_sentence, task='SA', few_shot=True):
    if task=='SA':
        defination='Introduction:\nPlease determine whether the last given sentence is a positive or negative statement by following the demonstrations below, only give the polarity.\n'
        demonstration='''
Demonstration:
Given sentence:"a slick , engrossing melodrama .", Polarity:"positive"
Given sentence:"the drama discloses almost nothing .", Polarity:"negative"
        '''
        if not few_shot:
            demonstration=''
        inp=f"Given sentence:\"{given_sentence}\", Polarity:"
        return defination+demonstration+inp
    if task=='fever':
        defination='Introdution:\nDetermine the relationship between the following given claim and evidence, specifically whether it is "SUPPORTS" or "REFUTES.", only give the label.'      
        demonstration='''
Demonstration:
Evidence:"As the Vietnam War raged in 1969 , Yoko Ono and her husband John Lennon held two week-long Bed-Ins for Peace , one at the Hilton Hotel in Amsterdam and one at Fairmont The Queen Elizabeth in Montreal , each of which were intended to be non-violent protests against wars , and experimental tests of new ways to promote peace . Vietnam War Vietnam War Yoko Ono Yoko Ono John Lennon John Lennon Hilton Hotel Hilton Amsterdam Amsterdam Amsterdam Montreal Montreal"
Claim:"As the Vietnam War raged in 1969, Yoko Ono and her husband John Lennon did not have two week-long Bed-Ins for Peace."
label:"REFUTES"

Evidence:"His most commercially successful role to date has been that of Remus Lupin in the Harry Potter film series . Remus Lupin Remus Lupin"
Claim:"There have been many notable performances by David Thewlis."
label:"SUPPORTS"\n
        '''
        if not few_shot:
            demonstration=''  
        inp=f"{given_sentence}\nlabel:"
        return defination+demonstration+inp

def revise_sent_prompt(given_sentence, task='SA', label='positive', few_shot=True):
    if task=='SA':
        defination=f'Introduction:The following is a {label} statement, please modify its wording, while preserving its polarity, i.e., positive and negative, and its semantics as much as possible by following the demonstrations below.'
        if label=='positive':
            demonstration='''
Demonstration:
Given sentence:"it has more than a few moments that are insightful enough to be fondly remembered in the endlessly challenging maze of moviegoing .", Revised sentence:"It offers numerous insightful moments that linger fondly in the ever-changing landscape of cinema, adding depth to the moviegoing experience."
            '''
        elif label=='negative':
            demonstration='''
Demonstration:
Given sentence:"i did n't find much fascination in the swinging .", Revised sentence:"I found little fascination in the swinging.
            '''
        else:
            assert 0==1
        inp=f"Given sentence:\"{given_sentence}\", Revised sentence:"
        if not few_shot:
            demonstration=''
        return defination+demonstration+inp
    if task=='fever':
        if label=='SUPPORTS':
            defination=f'Introdution: The following is a Claim and an Evidence. Please write a new Claim by following the demonstrations below, while making sure the evidence can still {label.lower()} the claim.\n'
            demonstration='''
Demonstration:
Evidence:"His most commercially successful role to date has been that of Remus Lupin in the Harry Potter film series . Remus Lupin Remus Lupin"
Claim:"There have been many notable performances by David Thewlis."
Revised Claim:"David Thewlis has delivered numerous remarkable performances."\n
            '''
            inp=f"{given_sentence}\nRevised Claim:"
        elif label=='REFUTES':
            defination=f'Introdution: The following is a Claim and an Evidence. Please write a new Claim by following the demonstrations below, while making sure your Claim will confilt with the fact given by the evidence.\n'
            demonstration='''
Demonstration:
Evidence:"Bali is located at the westernmost end of the Lesser Sunda Islands , between Java to the west and Lombok to the east ."
Claim:"Bali is not between Java and Lombok."
Revised Claim:"Bali is located at the easternmost end of the Lesser Sunda Islands."

Evidence:"William Bradley Pitt -LRB- born December 18 , 1963 -RRB- is an American actor and producer ."
Claim:"Brad Pitt is not an actor."
Revised Claim:"Brad Pitt is primarily known for his music career."\n\n'''
            inp=f"{given_sentence}\nRevised Claim:"
        if not few_shot:
            demonstration=''
        return defination+demonstration+inp