import pandas as pd
import os
import os.path as osp
# 取票数最高的结果
from collections import Counter

def read_csv(file_path):
    df = pd.read_csv(file_path,delimiter='\t',encoding='utf-8')
    print(df)
    return df


def merge_dir_csv(store_dir):
    list_dir =os.listdir(store_dir)
    store_data = []
    count = 0
    for file_path in [iter for iter in list_dir if iter.find('csv')!=-1]:
        data = read_csv(osp.join(store_dir,file_path))

        data.rename(columns={'label':'label{}'.format(count)},inplace=True)
        if count > 0:
            data = data['label{}'.format(count)]
        store_data.append(data)
        count +=1

    merged_data = pd.concat(store_data, axis=1)
    merged_data['label'] = merged_data.apply(lambda x: Counter([x.label0, x.label1, x.label2]).most_common()[0][0], axis=1)

    result = merged_data[['id','label']]
    return  result


if __name__ == '__main__':
    result = merge_dir_csv('./result/')
    result.to_csv('./result/merged_result.csv',encoding='utf-8',sep='\t',index=False)
