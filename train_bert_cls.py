import pandas as pd
import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from Data import ContentDataSet
from Bert_CLS import SentenceClassffier
from sklearn.metrics import classification_report
import logging
import argparse
from Data import convert_input_to_tensor
from functools import partial
from transformers import BertTokenizer,AdamW,get_linear_schedule_with_warmup,BertConfig
import os
import os.path as osp
from FGM import FGM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',type = int,default=5)
    parser.add_argument('--train_path',type = str,default='./dataset/train.csv')
    parser.add_argument('--epochs',type= int,default=5)
    parser.add_argument('--learning_rate',type=float,default=1e-5)
    parser.add_argument('--warm_up_rate',type=float,default=0.01)
    parser.add_argument('--is_cv',action='store_true')
    parser.add_argument('--cv_number',type= int ,default=1)
    parser.add_argument('--test_path',type=str,default='./dataset/test1.csv')
    parser.add_argument('--cache_dir',type= str,default='./cache')
    parser.add_argument('--save_dir',type=str,default='./model_save')
    parser.add_argument('--out_dir',type=str,default='./result/')
    parser.add_argument('--pretrain_model',type=str,default='./pretrained_model')
    parser.add_argument('--pooling',type=str,default='first-last-avg')
    parser.add_argument('--label_numbers',type=int,default=2)
    parser.add_argument('--log_interval',type=int,default=10)
    parser.add_argument('--inference_model',type=str,default='./model_save/best_model_for_cv0')
    parser.add_argument('--inference',action='store_true')
    args = parser.parse_args()
    logger.info(args)
    return args


def eval(model,dev_loader,device):
    model.eval()
    losses = []
    y_pred = []
    y_true = []
    for batch_idx,batch_data in enumerate(dev_loader):
        input_ids = batch_data['input_ids'].squeeze(1).to(device)
        attention_mask = batch_data['attention_mask'].squeeze(1).to(device)
        token_type_ids = batch_data['token_type_ids'].squeeze(1).to(device)
        batch_labels = batch_data['label']
        y_true.extend(batch_labels)

        batch_labels = batch_labels.to(device)
        loss, logits = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                        labels=batch_labels)
        losses.append(loss.item())
        softmax = nn.Softmax(dim=-1)
        logits = softmax(logits)
        pred_labels = torch.argmax(logits,dim=-1)
        pred_labels = pred_labels.cpu().tolist()
        y_pred.extend(pred_labels)
    result = classification_report(y_true,y_pred,output_dict=True)
    logger.info(result)
    metrics_dict = {}
    metrics_dict['loss'] = sum(losses)/len(losses)
    metrics_dict['accuracy'] = result['accuracy']
    metrics_dict['macro_avg_f1'] = result['macro avg']['f1-score']
    metrics_dict['weighted_avg_f1'] = result['weighted avg']['f1-score']
    return metrics_dict



def parser_for_train(args):
    config = BertConfig.from_pretrained(pretrained_model_name_or_path=args.pretrain_model,cache_dir = args.cache_dir)
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=args.pretrain_model,cache_dir=args.cache_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    convert_fn = partial(
        convert_input_to_tensor,
        tokenizer = tokenizer,
        max_length = config.max_position_embeddings
    )
    total_cv_number = args.cv_number if args.is_cv else 1
    for current_cv in range(total_cv_number):
        logger.info('current_cv:{} starting '.format(current_cv))

        model = SentenceClassffier(pretrain_model=args.pretrain_model, cache_dir=args.cache_dir,
                                   pooling=args.pooling,
                                   label_number=args.label_numbers)
        model = model.to(device)
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        dev_dataset = ContentDataSet(args.train_path, mode='dev', is_cv=args.is_cv, cv_number=args.cv_number,
                                     current_k=current_cv, trun_func=convert_fn)
        train_dataset = ContentDataSet(args.train_path, mode='train', is_cv=args.is_cv, cv_number=args.cv_number,
                                       current_k=current_cv, trun_func=convert_fn)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataloader) * args.epochs * args.warm_up_rate, num_training_steps=len(train_dataloader) * args.epochs)


        best_f1 = 0.0

        # for fgm
        # fgm = FGM(model)
        for epoch in range(args.epochs):
            model.train()
            losses = []
            for batch_idx, batch_data in enumerate(train_dataloader):
                input_ids = batch_data['input_ids'].squeeze(1).to(device)
                attention_mask = batch_data['attention_mask'].squeeze(1).to(device)
                token_type_ids = batch_data['token_type_ids'].squeeze(1).to(device)
                batch_labels = batch_data['label'].to(device)
                loss, _ = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                labels=batch_labels)
                optimizer.zero_grad()
                loss.backward()
                # ###fgm insert
                # fgm.attack()
                # loss_adv, _ = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                #                 labels=batch_labels)
                # loss_adv.backward()
                # fgm.restore()
                # ### fgm ending
                optimizer.step()
                scheduler.step()

                losses.append(loss.item())
                if batch_idx %args.log_interval ==0 :
                    logger.info('current_cv:{},current_epoch:{},current_step:{},current_step_loss:{}'.format(current_cv,epoch,batch_idx,loss.item()))

            logger.info('current_cv:{},current_epoch:{},current_epoch_loss:{}'.format(current_cv,epoch,sum(losses)/len(losses)))
            metrics = eval(model,dev_dataloader,device)
            current_f1 = metrics['weighted_avg_f1']
            if current_f1 > best_f1:
                best_f1 = current_f1
                torch.save(model.state_dict(),osp.join(args.save_dir,'best_model_for_cv{}'.format(current_cv)))

    logger.info('current_cv:{} ending'.format(current_cv))


def parser_for_inference(args):
    config = BertConfig.from_pretrained(pretrained_model_name_or_path=args.pretrain_model, cache_dir=args.cache_dir)
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=args.pretrain_model,
                                              cache_dir=args.cache_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    convert_fn = partial(
        convert_input_to_tensor,
        tokenizer=tokenizer,
        max_length=config.max_position_embeddings
    )
    model = SentenceClassffier(pretrain_model=args.pretrain_model, cache_dir=args.cache_dir,
                                pooling=args.pooling,
                                label_number=args.label_numbers)

    state_dict = {k.replace('module.', ''):v for k, v in torch.load(args.inference_model,map_location=device).items()}
    model.load_state_dict(state_dict)

    model = model.to(device)
    test_dataset = ContentDataSet(args.test_path,is_cv = False,mode ='test',trun_func=convert_fn)
    test_dataloader = DataLoader(test_dataset,shuffle=False,batch_size=args.batch_size)

    model.eval()

    ids = []
    labels = []
    for batch_idx,batch_data in enumerate(test_dataloader):
        if batch_idx % args.log_interval ==0:
            logger.info('current process :{}'.format(batch_idx/len(test_dataloader)))

        input_ids = batch_data['input_ids'].squeeze(1).to(device)
        attention_mask = batch_data['attention_mask'].squeeze(1).to(device)
        token_type_ids = batch_data['token_type_ids'].squeeze(1).to(device)
        batch_ids = batch_data['id'].cpu().tolist()

        ids.extend(batch_ids)
        logits = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                labels=None)
        softmax = nn.Softmax(dim=-1)
        logits = softmax(logits)
        pred_labels = torch.argmax(logits, dim=-1)
        pred_labels = pred_labels.cpu().tolist()
        labels.extend(pred_labels)

    df = pd.DataFrame({'id':ids,'label':labels})
    df.to_csv(osp.join(args.out_dir,'result.csv'),encoding='utf-8',index=False,sep='\t')


if __name__== '__main__':
    args = get_args()
    if args.inference:
        parser_for_inference(args)
    else :
        parser_for_train(args)
