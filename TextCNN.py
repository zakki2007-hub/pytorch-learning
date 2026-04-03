# -*- coding: gbk -*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        # 卷积核数量*卷积核尺寸数量，计算总特征数
        self.num_filter_tool = num_fliters * len(filter_sizes) 
        # 词向量表
        self.W = nn.Embedding(vocab_size, embedding_size)
        # 分类权重
        self.Weight = nn.Linear(self.num_filter_tool, num_classes, bias=False) 
        # 手动定义偏置项
        self.Bias = nn.Parameter(torch.ones([num_classes])) 
        # 定义一组卷积层列表，卷积核尺寸为filter_sizes，卷积核数量为num_fliters
        self.fliter_list = nn.ModuleList([nn.Conv2d(1, num_fliters, (size, embedding_size)) for size in filter_sizes])

    def forward(self, X):
        # 将输入的索引X转化为词向量矩阵
        embedded_chars = self.W(X)
        # 在第一维插入一个维度1
        # 理由：Conv2d的输入要求是4维的，分别是batch_size, channel, height, width
        embedded_chars = embedded_chars.unsqueeze(1)

        pooled_outputs = []
        for i, conv in enumerate(self.fliter_list):
            # 卷积+激活，relu函数去掉无意义的负值
            h = F.relu(conv(embedded_chars))
            # 最大池化，将整个卷积结果最大值挑出来
            mp = nn.MaxPool2d((sequence_length - filter_sizes[i] + 1, 1))

            pooled = mp(h)
            pooled_outputs.append(pooled)
        
        # 将不同卷积核尺寸的卷积结果拼接在一起
        h_pool = torch.cat(pooled_outputs, len(filter_sizes))
        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filter_tool])
        model = self.Weight(h_pool_flat) + self.Bias
        return model
    
if __name__ == "__main__":
    embedding_size = 2
    sequence_length = 3
    num_classes = 2
    filter_sizes = [2, 2, 2]
    num_fliters = 3

    # 3个单词的句子
    sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
    labels = [1, 1, 1, 0, 0, 0] # 1表示正面情感，0表示负面情感

    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    vocab_size = len(word_dict)

    model = TextCNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    inputs = torch.LongTensor([[word_dict[n] for n in s.split()] for s in sentences])
    targets = torch.LongTensor([out for out in labels])

    for epoch in range(5000):
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch: {epoch + 1}, Loss: {loss.item():.4f}")
        
        loss.backward()
        optimizer.step()
    # 测试模型
    test_text = "he you that"
    tests = [np.asarray([word_dict[n] for n in test_text.split()])]
    test_batch = torch.LongTensor(tests)
    
    predict = model(test_batch).data.max(1, keepdim=True)[1]
    
    if predict[0][0] == 1:
        print(test_text, "正面情感")
    else:        
        print(test_text, "负面情感")