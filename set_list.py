import pickle
import random


def write_file(filename, str):
    """
    写入文件
    :param str: 字符串
    :return: 无
    """
    writefile = open("./data/" + filename, 'a+', encoding='utf-8')
    writefile.write(str + '\n')
    writefile.close()


def load_pkl(filename):
    with open('./pkl_save/' + filename, 'rb', ) as file:
        return pickle.load(file)


texts = load_pkl('texts.pickle')
# print(len(list(set(texts))))
unique_texts = list(set(texts))

# print(unique_texts)

random.shuffle(unique_texts)
for i in range(len(unique_texts) - 1):
    write_file('context.txt', unique_texts[i].strip() + unique_texts[i + 1].strip())
    write_file('texts.txt', unique_texts[i].strip() + '|' + unique_texts[i + 1].strip())

random.shuffle(unique_texts)
for i in range(len(unique_texts) - 2):
    write_file('context.txt', unique_texts[i].strip() + unique_texts[i + 1].strip() + unique_texts[i + 2].strip())
    write_file('texts.txt',
               unique_texts[i].strip() + '|' + unique_texts[i + 1].strip() + '|' + unique_texts[i + 2].strip())

random.shuffle(unique_texts)
for i in range(len(unique_texts) - 3):
    write_file('context.txt',
               unique_texts[i].strip() + unique_texts[i + 1].strip() + unique_texts[i + 2].strip() + unique_texts[
                   i + 3].strip())
    write_file('texts.txt',
               unique_texts[i].strip() + '|' + unique_texts[i + 1].strip() + '|' + unique_texts[i + 2].strip() + '|' +
               unique_texts[i + 3].strip())
