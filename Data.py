import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from transformers import BertTokenizer

def read_csv(file_path):
    df = pd.read_csv(file_path,delimiter='\t')
    return df


class ContentDataSet(Dataset):
    def __init__(self,file_path,mode ='train',is_cv = False,cv_number=5,current_k=0,trun_func=None):
            self.data = read_csv(file_path)
            self.texts = self.data['text'].tolist()
            self.mode = mode
            self.current_k = current_k
            self.trun_func = trun_func
            self.ids = self.data['id'].tolist()
            if self.mode !='test':
                self.labels = self.data['label'].tolist()

            if  is_cv and cv_number > 0:
                fold_length = len(self.texts) // cv_number
                fold_sizes = [0]+[fold_length]*(cv_number-1)
                fold_sizes = np.cumsum(fold_sizes)
                if self.current_k == (cv_number-1):

                    if self.mode == 'train':
                        self.texts = self.texts[0:fold_sizes[current_k]]
                        self.labels = self.labels[0:fold_sizes[current_k]]

                    if self.mode =='dev':
                        self.texts = self.texts[fold_sizes[current_k]:]
                        self.labels = self.labels[fold_sizes[current_k]:]
                else:
                    if self.mode =='train':
                        self.texts =self.texts[0:fold_sizes[current_k]]+self.texts[fold_sizes[current_k+1]:]
                        self.labels = self.labels[0:fold_sizes[current_k]]+self.labels[fold_sizes[current_k+1]:]

                    if self.mode =='dev':
                        self.labels = self.labels[fold_sizes[current_k]:fold_sizes[current_k+1]]
                        self.texts = self.texts[fold_sizes[current_k]:fold_sizes[current_k + 1]]


    def __getitem__(self, item):
        example = {}
        text = self.texts[item]
        id = self.ids[item]
        example['text'] =text
        example['id'] = id
        if self.mode !='test':
            label = self.labels[item]
            example['label'] = label
        if self.trun_func:
            return self.trun_func(example)
        return example

    def __len__(self):
        return len(self.texts)

def convert_input_to_tensor(example,tokenizer,max_length):

    text = example['text']
    encoded_text = tokenizer(text,max_length =max_length,return_tensors ='pt',truncation =True,padding ='max_length')
    sample = {}
    sample['input_ids'] = encoded_text['input_ids']
    sample['attention_mask'] = encoded_text['attention_mask']
    sample['token_type_ids'] = encoded_text['token_type_ids']
    sample['id'] = example['id']
    if 'label' in example.keys():
        label = torch.tensor(example['label'], dtype=torch.long)
        sample['label'] = label
    return sample



if __name__=='__main__':
    numbers = [0]+[5]*12
    print(np.cumsum(numbers))