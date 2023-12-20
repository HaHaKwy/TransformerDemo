from torch.utils.data import DataLoader

from transformer import TransformerDecoder, TransformerEncoder

import torch
import utils

from MyDataset import *
from dataProgress import *
from train_predict import *


device_ids = [1, 2, 3]

num_hiddens = 32
num_layers = 2
dropout = 0.1
batch_size = 100
num_steps = 200

lr = 0.005
num_epochs = 150

# device = utils.try_gpu()
device = torch.device('cuda:3')

ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
key_size, query_size, value_size = 32, 32, 32
norm_shape = [32]

if __name__ == '__main__':
    #加载数据
    code_path = './data/JavaData/java_code.txt'
    comment_path = './data/JavaData/comment.txt'

    # word <-> id
    code, comment = creat_dataset(code_path, comment_path)
    code_word2id, code_id2word = build_dict(build_cropus(code), 2)
    comment_word2id, comment_id2word = build_dict(build_cropus(comment), 2)

    # id列表
    comment_tensor = build_tensor(comment, comment_word2id, num_steps)
    code_tensor = build_tensor(code, code_word2id, num_steps)
    comment_tensor = np.array(comment_tensor)
    code_tensor = np.array(code_tensor)

    train_comment_tensor = comment_tensor[:50000]
    test_comment_tensor = comment_tensor[50000:]
    train_code_tensor = code_tensor[:50000]
    test_code_tensor = code_tensor[50000:]


    # Dataset Dataloader
    train_batch_num = train_code_tensor.shape[0] // batch_size
    train_dataset = MyDataset(train_comment_tensor,train_code_tensor)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last = True)

    print("Successfully load data")
    print(f"Total batch num: {len(train_loader)} || {train_batch_num}")

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
    # net = torch.nn.DataParallel(net, device_ids=device_ids,dim=0)

    # 训练
    print("Start train model")
    train_seq2seq(net, train_loader, lr, num_epochs, code_word2id, device)

    torch.save(net.state_dict(), './modules_saved/CodeGen_Params.pth')

    # 简单预测
    doc = 'generate a function to print \' Hello world ! \''
    code, _ = predict_seq2seq(net, doc, comment_word2id, code_word2id, num_steps, device, False)

    print("Generation:")
    print(code)
