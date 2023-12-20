import time
import torch
from torch import nn
import utils
import dataProgress

"""
训练函数、预测函数、损失函数
"""

# 自定义交叉熵损失函数 乘以掩码
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self, pred, label, valid_len):
        # pred: [batch_size,num_steps,vocab_size]
        # label:[batch_size,num_steps]
        # valid:[batch_size]
        weights = torch.ones_like(label)
        weights = utils.sequence_mask(weights, valid_len)
        self.reduction = 'none'

        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)
        # 注意顺序 正常交叉熵损失比较的是（batch_size,各分类的得分,N）和（batch_size,实际类别,N）
        # 这里N=num_steps

        weighted_loss = (unweighted_loss * weights)
        weighted_loss = weighted_loss.mean(dim = 1)
        return weighted_loss


def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr = lr)

    # optimizer = nn.DataParallel(optimizer, device_ids=device_ids)
    loss = MaskedSoftmaxCELoss()

    net.train()
    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"Epoch {epoch + 1} train starts...")
        metric = utils.Accumulator(2)
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]

            bos = torch.tensor([tgt_vocab['<end>']] * Y.shape[0], device = device).reshape(-1, 1)

            dec_input = torch.cat([bos, Y[:, :-1]], dim = 1)

            Y_hat = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()

            utils.grad_clipping(net, 1)  # 梯度裁剪
            optimizer.step()

            num_tokens = Y_valid_len.sum()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
            if (i + 1) % 50 == 0:
                print(f"{i + 1}th batch finished.")

        end_time = time.time()
        print(f'{epoch+1} cost {end_time-start_time} secs')
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch: {epoch + 1} , avg loss: {metric[0] / metric[1]}")
            torch.save(net.state_dict(), f'./modules_saved/CodeGen_Params_{epoch + 1}.pth')

    print("===== 训练结束 =====")
    print(f'Train loss {metric[0] / metric[1]:.3f}')


def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, device, save_attention_weights = False):
    net.eval()
    # 处理输入
    src_word_list = src_sentence.lower().split(' ')
    src_word_list = [x for x in src_word_list if x!='']

    src_tokens = [src_vocab[x] for x in src_word_list if x in src_vocab.keys()]  + [src_vocab['<end>']] 
    # src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<end>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device = device)
    src_tokens = dataProgress.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])

    # 添加批量这一维度 [num_steps] => [1,num_steps]
    enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype = torch.long, device = device), dim = 0)

    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 添加批量维度
    # 解码器第一个时间步的输入为<bos> 结合编码器输入的状态 之后不断地向dec_X中添加已经预测得到的输入
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<end>']], dtype = torch.long, device = device), dim = 0)
    output_seq, attention_weights_seq = [], []

    # hidden_state = dec_state
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # X_and_state = torch.cat((dec_X,hidden_state),dim=2)
        # Y,hidden_state = net.decoder(X_and_state,dec_state)

        # 可能性最高的词元作为解码器下一时间步的输入
        # Y: [batch_size, num_steps, vocab_size] / [1,1,vocab_size]
        dec_X = Y.argmax(dim = 2)
        # dec_X: [batch_size, num_steps] / [1,1]
        pred = dec_X.squeeze(dim = 0).type(torch.int32).item()
        if save_attention_weights:
            attention_weights_seq.append(net.decoder.attention_weights)
        if pred == tgt_vocab['<end>']:
            break
        output_seq.append(pred)
    
    return output_seq, attention_weights_seq
    # return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weights_seq