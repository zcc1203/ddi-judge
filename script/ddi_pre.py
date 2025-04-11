import openai
from openai import OpenAI
import random
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score,accuracy_score
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit import Chem
import warnings
from rdkit import RDLogger
from steamship import Steamship
import datetime
import os 
from sklearn.model_selection import KFold
key=''


client = OpenAI(
    base_url="",
    api_key=key
)

EVENT_NUM = 65
CV=5
SEED = 0
ANSWER=10

def get_index(label_matrix, event_num, seed, CV):
    index_all_class = np.zeros(len(label_matrix))
    for j in range(event_num):
        index = np.where(label_matrix == j)
        kf = KFold(n_splits=CV, shuffle=True, random_state=seed)
        k_num = 0
        for train_index, test_index in kf.split(range(len(index[0]))):
            index_all_class[index[0][test_index]] = k_num
            k_num += 1

    return index_all_class

def generate_response_by_gpt35_openai(prompt,model_engine='gpt-3.5-turbo'):
    
    completion = client.chat.completions.create(
        model=model_engine, temperature=1, n=ANSWER, 
        messages=[{"role": "user", "content": prompt}],
    )
    message = completion.choices
    message = [i.message.content.strip() for i in message]
    return message    
def generate_response_by_gpt4_openai(prompt,model_engine='gpt-4o'):
    completion = client.chat.completions.create(
        model=model_engine, temperature=1, n=ANSWER, 
        messages=[{"role": "user", "content": prompt}],
    )
    message = completion.choices
    message = [i.message.content.strip() for i in message]
    #import pdb;pdb.set_trace()
    return message

def create_base_ddi_prompt(events):
    message=f'You are an experienced pharmacologist. Your task is to determine the type of interaction between two drugs. You can use available pharmacological data, metabolic pathways, and known interaction mechanisms. Do not rely solely on chemical structures but consider the pharmacodynamics, metabolism via enzymes, receptor interactions, and any relevant historical clinical data or known interaction cases. Given the names and SMILES structures of known drug A and drug B, the question is: Is there an adverse interaction between the two drugs? Answer "yes" if there is an adverse interaction, and answer "no" if there is no adverse interaction.\n'
    message+=f'You must only answer the prediction, yes or no, Do not include explanations about limitations or the need for additional data. Do not say sorry. \n'
    message+=f'Additional Instructions:\n Consider known metabolic interactions, including enzyme inhibition or induction, and receptor interactions. \n Utilize clinical data, such as FDA labels, interaction databases, and peer-reviewed studies, to support your prediction.\n'
    print(message)
    return message
def create_prompt_zero_shot(mes,example):

    druganame = example[1]
    drugasmile=example[3]
    drugbname=example[4]
    drugbsmile=example[6]
    message = mes;
    message+=f'Templates are provided in the beginning.\n'
    message+=f'Drug A Name: Drug A Name \n'
    message+=f'Drug A Smiles: Smiles structure of Drug A\n'
    message+=f'Drug B Name: Drug B Name \n'
    message+=f'Drug B Smiles: Smiles structure of Drug B\n'
    message+=f'adverse： yes\n'
    message+=f'Drug A Name: Drug A Name \n'
    message+=f'Drug A Smiles: Smiles structure of Drug A\n'
    message+=f'Drug B Name: Drug B Name \n'
    message+=f'Drug B Smiles: Smiles structure of Drug B\n'
    message+=f'adverse： no\n'
    message+=f'Drug A Name: {druganame}\n'
    message+=f'Drug A Smiles: {drugasmile}\n'
    message+=f'Drug B Name: {drugbname}\n'
    message+=f'Drug B Smiles: {drugbsmile}\n'
    message+=f'adverse：\n'
    print(message)
    return message

random.seed(42) 
#read bace dataset
ddi = pd.read_csv("../data/ddi/data.csv")
ddi_values =ddi.values
labels = ddi['label']
labels =np.array(labels)
index_all_class = get_index(labels, EVENT_NUM, SEED, CV)
LABEL = []
with open('../data/ddi/label.txt') as file:
    for line in file:
        print(line.strip())
        LABEL.append(line.strip())


message = create_base_ddi_prompt(LABEL)
dst_path = f'../res/ddi/'

for k in range(CV):
    dst_csv_name = f'ddi_test_{k}.csv'
    train_index = np.where(index_all_class != k)
    test_index = np.where(index_all_class == k)
    ddi_train = ddi.iloc[train_index]
    ddi_test = ddi.iloc[test_index]
    ###ddi_test DataFrame
    #test_res_dict ={}
    pred_dict = {f'pred{i}': [] for i in range(0, ANSWER+1)}
    sample_size = 100
    ddi_test_sa= ddi_test.sample(sample_size)
    for dt in ddi_test_sa.values:
        prompt = create_prompt_zero_shot(message,dt)
        res = generate_response_by_gpt4_openai(prompt)
        pred_dict[f'pred0'].append(dt[8])
        for i in range(ANSWER):
            #import pdb;pdb.set_trace()
            if 'no' in res[i] or 'NO' in res[i] or 'No' in res[i]:
                r0=0
            elif 'yes' in res[i] or 'Yes' in res[i] or 'YES' in res[i]:
                r0 =1
            else:
                import pdb;pdb.set_trace()
                print("ERROR:",res[i])
            print(r0)
            pred_dict[f'pred{i+1}'] .append(r0)
    #import pdb;pdb.set_trace()
    new_df = pd.DataFrame(pred_dict)
    new_df = new_df.rename(columns={'pred0':'label'})
    #import pdb;pdb.set_trace()
    #dfaf = pd.concat([ddi_test, new_df], axis=1)
    new_df.to_csv(dst_path+dst_csv_name)
    
    
    
    
import pdb;pdb.set_trace()