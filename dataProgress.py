import numpy as np


def creat_dataset(a, b):
    # a : code
    # b: comment
    with open(a, encoding = 'utf-8') as tc:
        lines1 = tc.readlines()
        for i in range(len(lines1)):
            lines1[i] = "<start> " + lines1[i].strip('\n') + " <end>"

    with open(b, encoding = 'utf-8') as ts:
        lines2 = ts.readlines()
        for i in range(len(lines2)):
            lines2[i] = "<start> " + lines2[i].strip('\n') + " <end>"

    if (len(lines1) != len(lines2)):
        print("数据量不匹配")
    return lines1, lines2


# tokenize
def build_cropus(data):
    crpous = []
    for i in range(len(data)):
        cr = data[i].strip().lower()
        cr = cr.split()
        crpous.extend(cr)
    return crpous

def truncate_pad(line, num_steps, padding_token):
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))

# 构造词典，统计每个词的频率，并根据频率将每个词转换为一个整数id
def build_dict(corpus, frequency):
    # 首先统计每个不同词的频率（出现的次数），使用一个词典记录
    word_freq_dict = dict()
    for word in corpus:
        if word not in word_freq_dict:
            word_freq_dict[word] = 0
        word_freq_dict[word] += 1

    # 将这个词典中的词，按照出现次数排序，出现次数越高，排序越靠前
    word_freq_dict = sorted(word_freq_dict.items(), key = lambda x: x[1], reverse = True)

    word2id_dict = {'<pad>': 0, '<unk>': 1}
    id2word_dict = {0: '<pad>', 1: '<unk>'}

    # 按照频率，从高到低，开始遍历每个单词，并为这个单词构造一个独一无二的id
    for word, freq in word_freq_dict:
        if freq > frequency:
            curr_id = len(word2id_dict)
            word2id_dict[word] = curr_id
            id2word_dict[curr_id] = word
        else:
            word2id_dict[word] = 1  # <unk>
    return word2id_dict, id2word_dict


# 向量化 token列表->id列表
def build_tensor(data, dicta, maxlen):
    tensor = []
    for i in range(len(data)):
        subtensor = []
        lista = data[i].split()
        for j in range(len(lista)):
            index = dicta.get(lista[j])
            subtensor.append(index)

        if len(subtensor) < maxlen:
            subtensor += [0] * (maxlen - len(subtensor))  # <pad>
        else:
            subtensor = subtensor[:maxlen]

        tensor.append(subtensor)
    return tensor


# code_path = './data/JavaData/java_code.txt'
# comment_path = './data/JavaData/comment.txt'
#
# code, comment = creat_dataset(code_path, comment_path)
# code_word2id, code_id2word = build_dict(build_cropus(code), 2)
# comment_word2id, comment_id2word = build_dict(build_cropus(comment), 2)
#
# code_maxlen = 200
# comment_maxlen = 30
# code_vocab_size = len(code_id2word)
# comment_vocab_size = len(comment_id2word)
#
# code_tensor = build_tensor(code, code_word2id, code_maxlen)  # id列表
# comment_tensor = build_tensor(comment, comment_word2id, comment_maxlen)  # id列表
# code_tensor = np.array(code_tensor)
# comment_tensor = np.array(comment_tensor)
#
# test_code_tensor = code_tensor[:20000]
# train_code_tensor = code_tensor[20000:]
#
# test_comment_tensor = comment_tensor[:20000]
# train_comment_tensor = comment_tensor[20000:]

