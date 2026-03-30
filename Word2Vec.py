# -*- coding: gbk -*-
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


def random_batch():
    random_inputs = [] # 输入词
    random_labels = [] # 上下文词
    random_index = np.random.choice(range(len(skip_grams)), batch_size, replace=False) # 从数据集中随机选择batch_size个样本的索引，replace=False表示不放回抽样

    for i in random_index:
        random_inputs.append([skip_grams[i][0]]) # 将输入的单词索引转化为一个列表，并添加到随机输入列表中
        random_labels.append(skip_grams[i][1]) # 将标签（预测的下一个词的索引）添加到随机标签列表中

    return random_inputs, random_labels

# 模型
class Word2Vec(nn.Module):
    def __init__(self):
        super(Word2Vec, self).__init__()
        self.W = nn.Embedding(voc_size, embedding_size)
        # 定义一个嵌入层，将输入的单词索引映射到嵌入空间，输入大小为voc_size，输出大小为embedding_size，bias=False表示不使用偏置项
        self.WT = nn.Linear(embedding_size, voc_size, bias=False)
        # 定义另一个线性层，将嵌入空间的向量映射回词汇表大小的空间，输入大小为embedding_size，输出大小为voc_size，bias=False表示不使用偏置项
    def forward(self, X):
        # X: [batch_size]
        hidden_layer = self.W(X)
         # 将输入的单词索引通过嵌入层映射到嵌入空间，得到隐藏层的输出
        output_layer = self.WT(hidden_layer) 
        # 将隐藏层的输出通过第二个线性层映射回词汇表大小的空间，得到输出层的输出
        return output_layer

if __name__ == "__main__":
    batch_size = 16
    embedding_size = 100
    with open("train.txt", "r", encoding="gbk") as f:
        text = f.read()
        # 简单分句（按点号或换行分）
        sentences = text.split('\n')
        # 只要有内容的句子
        sentences = [s.strip() for s in sentences if len(s) > 10]    
    word_sequence = " ".join(sentences).split()
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list)) # 去重，得到词汇表
    word_dict = {w: i for i, w in enumerate(word_list)} # 创建一个字典，将每个单词映射到一个唯一的索引
    voc_size = len(word_list) # 词汇表大小

    skip_grams = []
    for i in range(1, len(word_sequence) -1 ):
        target = word_dict[word_sequence[i]] # 目标词的索引
        context = [word_dict[word_sequence[i-1]], word_dict[word_sequence[i+1]]] # 上下文词的索引
        for w in context:
            skip_grams.append([target, w]) # 将目标词和上下文词的索引作为一个样本添加到skip_grams列表中
    
    model = Word2Vec()
    criterion = nn.CrossEntropyLoss() # 定义损失函数为交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # 定义优化器

    for epoch in range(30000):
        input_batch, target_batch = random_batch() # 获取一个随机批次的输入和标签
        input_batch = torch.LongTensor(np.array(input_batch)).view(-1) # 将输入批次转换为PyTorch张量
        target_batch = torch.LongTensor(np.array(target_batch)) # 将标签批次转换为PyTorch长整型张量

        optimizer.zero_grad()
        output = model(input_batch) # 前向传播，得到模型的输出
        loss = criterion(output, target_batch.long()) # 计算损失
        
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss)) # 每1000个epoch打印一次损失值
        loss.backward() # 反向传播，计算梯度
        optimizer.step() # 更新模型参数

    for i, label in enumerate(word_list):
        W, WT = model.parameters()
        x, y = W[0][i].item(), W[1][i].item() # 获取词汇表中每个单词在嵌入空间中的坐标
        plt.scatter(x, y) # 在二维平面上绘制单词的坐标点
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom') # 在坐标点旁边添加单词标签
    plt.show() # 显示绘制的图形







