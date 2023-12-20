from torch.utils.data import Dataset

from dataProgress import *
import torch


class MyDataset(Dataset):
    def __init__(self, comment,code):
        super(MyDataset, self).__init__()
        self.code = code
        self.comment=comment

    def __getitem__(self, index):
        comment_array = self.comment[index]
        comment_valid = (comment_array != 0).sum()

        code_array = self.code[index]
        code_valid = (code_array != 0).sum()

        return comment_array,comment_valid,code_array,code_valid

    def __len__(self):
        return self.comment.shape[0]

