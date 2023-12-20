from transformer import TransformerDecoder, TransformerEncoder

import torch
import utils

from MyDataset import *
from dataProgress import *
from train_predict import *


# device_ids = [1, 2, 3]

num_hiddens = 32
num_layers = 2
dropout = 0.1
num_steps = 200

device = torch.device('cuda:3')

ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
key_size, query_size, value_size = 32, 32, 32
norm_shape = [32]


def array2code(array,id2word):
    code_str = ''
    for a in array:
        code_str+=id2word[a]
        code_str+=' '
    return code_str


if __name__ == '__main__':
    #加载数据
    code_path = './data/JavaData/java_code.txt'
    comment_path = './data/JavaData/comment.txt'

    # lines
    code, comment = creat_dataset(code_path, comment_path)
    
    # word <-> id
    code_word2id, code_id2word = build_dict(build_cropus(code), 2)
    comment_word2id, comment_id2word = build_dict(build_cropus(comment), 2)

    # 实例化模型
    encoder = TransformerEncoder(
        len(comment_word2id), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)
    decoder = TransformerDecoder(
        len(code_word2id), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)
    net = utils.EncoderDecoder(encoder, decoder)

    net.load_state_dict(torch.load('modules_saved/CodeGen_Params_150.pth'))
    net.to(device)
    print('Successfully loaded model...')

    while True:
        doc = input("Enter doc: ")
        if doc=='exit':
            break
        code_array, _ = predict_seq2seq(net, doc, comment_word2id, code_word2id, num_steps, device, False)
        print(f"Code pred: {array2code(code_array[1:],code_id2word)}")
        print('============================')
    

