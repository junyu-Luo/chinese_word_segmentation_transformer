import pandas as pd
from tqdm import trange
import pickle
import os
from collections import Counter

def write_file(filename,str):
    """
    写入文件
    :param str: 字符串
    :return: 无
    """
    writefile = open("./data/"+filename, 'a+',encoding='utf-8')
    writefile.write(str + '\n')
    writefile.close()


def save_pkl(filename, content):
    file = open('./pkl_save/' + filename, 'wb')
    pickle.dump(content, file)
    file.close()


def load_pkl(filename):
    with open('./pkl_save/' + filename, 'rb') as file:
        return pickle.load(file)


def get_list():
    if os.path.exists('./pkl_save/contexts.pickle'):
        contexts = load_pkl('contexts.pickle')
        icds = load_pkl('icds.pickle')
        texts = load_pkl('texts.pickle')
        norm_texts = load_pkl('norm_texts.pickle')
        print("读取数据完毕")
    else:
        contexts = []
        icds = []
        texts = []
        norm_texts = []
        print("读取文件中")
        data = pd.read_excel('./data/AI部研究-诊断数据准备-20181220.xlsx')
        print("读取数据中")
        for i in trange(len(data)):
            contexts.append(data['O_DIA_NAME'][i])
            icds.append(data['ICD六位码'][i])
            texts.append(data['NODENAME'][i])
            norm_texts.append(data['标准术语'][i])
        assert len(contexts) == len(icds) == len(texts) == len(norm_texts)
        print("读取数据完毕")
        save_pkl('contexts.pickle', contexts)
        save_pkl('icds.pickle', icds)
        save_pkl('texts.pickle', texts)
        save_pkl('norm_texts.pickle', norm_texts)
        print("保存读取数据完毕")
    return contexts, icds, texts, norm_texts


def get_chaos_lis():
    if os.path.exists('./pkl_save/chaos_list.pickle'):
        result = load_pkl('chaos_list.pickle')
    else:
        contexts, icds, texts, norm_texts = get_list()
        i = 0
        not_only = []
        while i < len(contexts):
            try:
                if contexts[i] != contexts[i + 1]:
                    not_only.append(contexts[i])
                i += 1
            except:
                not_only.append(contexts[i])
                break
        result = []
        context = []
        icd = []
        text = []
        norm_text = []
        i = 0
        assert contexts[0] == not_only[0]
        while True:
            if contexts[i] == not_only[0]:
                context.append(contexts[i])
                icd.append(icds[i])
                text.append(texts[i])
                norm_text.append(norm_texts[i])
                i += 1
                if i == len(contexts):
                    result.append([list(set(context)), text, norm_text, icd])
                    break

            else:
                result.append([list(set(context)), text, norm_text, icd])
                del not_only[0]
                context = []
                icd = []
                text = []
                norm_text = []
        save_pkl('chaos_list.pickle',result)
    return result


# print(len(chaos_lis))

def get_seq_list():
    if os.path.exists('./pkl_save/seq_list.pickle'):
        seq_list = load_pkl('seq_list.pickle')
    else:
        chaos_lis = get_chaos_lis()
        seq_list = []
        for content in chaos_lis:
            context = content[0][0]
            texts_lis = content[1]
            norm_texts_lis = content[2]
            icds_lis = content[3]
            falg = True
            for text in texts_lis:
                try:
                    context.index(text)
                except:
                    falg = False

            if falg:
                seq_index = []
                if len(texts_lis) !=1:
                    for text in texts_lis:
                        seq_index.append(context.index(text))
                else:
                    seq_index.append(context.index(texts_lis[0]))
                # print([context])
                seq_zip = list(zip(*sorted(list(zip(seq_index,texts_lis,norm_texts_lis,icds_lis)))))
                seq_text = list(seq_zip[1])
                seq_norm_text = list(seq_zip[2])
                seq_icd = list(seq_zip[3])
                # print(seq_text)
                # print(seq_norm_text)
                # print(seq_icd)
                seq_list.append([[context],seq_text,seq_norm_text,seq_icd])
    save_pkl('seq_list.pickle', seq_list)
    return seq_list

seq_list = get_seq_list()
# print(seq_list)




write_file('context.txt','<pad>')
write_file('texts.txt','<pad>')
write_file('norm_texts.txt', '<pad>')
write_file('icds.txt', '<pad>')

write_file('context.txt','<unk>')
write_file('texts.txt','<unk>')
write_file('norm_texts.txt', '<unk>')
write_file('icds.txt', '<unk>')

write_file('context.txt','<s>')
write_file('texts.txt','<s>')
write_file('norm_texts.txt', '<s>')
write_file('icds.txt', '<s>')

write_file('context.txt','</s>')
write_file('texts.txt','</s>')
write_file('norm_texts.txt', '</s>')
write_file('icds.txt', '</s>')

for content in seq_list:
    context = content[0][0]
    texts_lis = content[1]
    norm_texts_lis = content[2]
    icds_lis = content[3]
    write_file('context.txt',context.replace("\n","|"))
    write_file('texts.txt','|'.join(texts_lis).replace("\n","|"))
    write_file('norm_texts.txt','|'.join(norm_texts_lis))
    write_file('icds.txt','|'.join(icds_lis))




