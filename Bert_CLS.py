from transformers import BertModel,BertConfig
import torch.nn as nn
import torch

class SentenceClassffier(nn.Module):
    def __init__(self,pretrain_model,cache_dir,pooling = 'first-last-avg',label_number = 2):
        super(SentenceClassffier, self).__init__()
        self.config = BertConfig.from_pretrained(pretrain_model,cache_dir = cache_dir)
        self.bert = BertModel.from_pretrained(pretrain_model,config =self.config,cache_dir = cache_dir)
        self.pooling = pooling
        self.act = nn.LeakyReLU()
        self.label_number = label_number
        self.fn = nn.Linear(self.config.hidden_size,self.label_number)
        self.loss = nn.CrossEntropyLoss()
        ## todo
            ## 添加lstm层 捕捉上下文语义信息
            ### 或者使用cnn提取local信息


    def forward(self,text,attention_mask,token_type_ids,labels = None):
        out =self.bert(text,attention_mask = attention_mask,token_type_ids= token_type_ids,output_hidden_states = True)
        if self.pooling == 'cls':
            logits =  out.last_hidden_state[:, 0]  # [batch, 768]

        if self.pooling == 'pooler':
            logits =  out.pooler_output  # [batch, 768]

        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            logits =  torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]

        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            logits =  torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]

        if  labels is None:
            pred = self.act(self.fn(logits))
            return pred
        else:
            pred = self.act(self.fn(logits))
            loss = self.loss(pred,labels)
            return (loss,logits)