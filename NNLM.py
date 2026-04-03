# -*- coding: gbk -*-
import torch
import torch.nn as nn
import torch.optim as optim

def make_batch():
    input_batch = [] #存放输入特征（上下文词）
    target_batch = [] # 存放预测目标（预测的下一个词）

    for sen in sentences:

        word = sen.split()
        input = [word_dict[n] for n in word[:-1]]
        target = word_dict[word[-1]]

        input_batch.append(input)
        target_batch.append(target)

    return input_batch, target_batch

# Model
class NNLM(nn.Module): # 所有pyTorch的模型都必须继承nn.Module类
    def __init__(self): # 初始化，用于定义网络中的层和参数
        super(NNLM, self).__init__() # 调用父类nn.Module的初始化函数
        '''
        定义网络层和参数: 权重矩阵Weight和偏置项bias
        原论文公式 y = b+ Wx+ U*tanh(d + Hx)
        其中x是输入特征 y是输出预测 W、U、H是权重矩阵 b和d是偏置项
        '''
        # C：词向量矩阵（Lookup Table）：大小为[n_class, m]，其中n_class是词汇表的大小，m是词向量的维度 将输入的单词索引（数字）转化为m维的稠密向量（word embedding）
        self.C = nn.Embedding(n_class, m)
        # H: 隐藏层权重矩阵，线性层；
        # 输入维度是n_step * m（n_step个词，每个词是m维向量），输出维度是n_hidden（隐藏层的神经元数量），不使用偏置项
        # 对应公式里的H
        self.H = nn.Linear(n_step * m, n_hidden, bias=False)
        # d: 隐藏层的偏置项，使用nn.Parameter告诉pytorch这是一个需要被反向传播更新的参数，他是一个长度为n_hidden的向量，初始化为1，
        # 对应公式里面的d
        self.d = nn.Parameter(torch.ones(n_hidden))
        # U：隐藏层到输出层的权重矩阵，线形层，输入维度是隐藏层输出的hidden，输出维度是n_class（预测词表中的哪一个词，需要计算每个词的概率分数），不使用偏置项
        # 对应公式里的U
        self.U = nn.Linear(n_hidden, n_class, bias=False)
        # 输入层直连输出层的权重矩阵：NNLM的特色，不仅有通过隐藏层的连接，还有输入层直接连接输出层的权重矩阵
        # 输入维度是n_step * m，输出维度是n_class，不使用偏置项
        # 对应公式里的W
        self.W = nn.Linear(n_step * m, n_class, bias=False)
        # 输出层的偏置项，对应公式里的b，长度为n_class，初始化为1
        self.b = nn.Parameter(torch.ones(n_class))
    
    def forward(self, X): # 前向传播函数，将论文中的数学公式转化为计算过程
        '''
        输入X是一个大小为[batch_size, n_step]的张量，包含了每个样本的上下文词的索引
        例如 如果batch_size=2 n_step=3 那么X可能是[[0,1,2],[3,4,5]]，表示两个样本，每个样本有三个上下文词的索引
        '''
        # 将输入的单词索引转化为词向量，得到一个大小为[batch_size, n_step, m]的张量 数据从2维变成了3维
        X = self.C(X)
        # 将词向量张量展平为大小为[batch_size, n_step * m]的张量，以便输入到线性层中
        X = X.view(-1, n_step * m) # .view()函数将n_step个词的m维向量展平为一个n_step * m维的向量
        # PyTorch 中，-1 的意思是“自动计算这一维的大小”。在这里，我们固定了后面的维度是 n_step * m，PyTorch 会自动把第一维保持为 batch_size。这是一种非常安全且规范的写法。
        # 计算隐藏层的输出，先通过线性层H计算线性变换，然后加上偏置项d，最后通过tanh激活函数得到非线性输出
        tanh = torch.tanh(self.H(X) + self.d)
        # 计算输出层的分数，先通过线性层U计算隐藏层到输出层的变换，然后加上输入层直接连接输出层的变换（W）和输出层的偏置项b
        output = self.U(tanh) + self.W(X) + self.b # [batch_size, n_class]
        # 有两条路通向最终输出：
        # 第一条路（深层逻辑）：self.U(tanh)。把刚才经过激活函数的隐藏层特征，映射到整个词表大小（n_class）。
        # 第二条路（捷径/Skip Connection）： self.W(X)。把第 2 步最原始拼接好的长向量X，不经过隐藏层，直接映射到词表大小。NNLM 作者认为这能保留最原始的输入特征。
        return output 

if __name__ == '__main__':
    '''
    步骤一：定义全局超变量
    '''
    n_step = 2 # 上下文词的数量
    n_hidden = 2 # 隐藏层神经元数量 256、512
    m = 2 # 词向量(嵌入)的维度 100-768
    '''
    步骤二：NLP数据预处理基础 - 构建词汇表
    '''
    sentences = ["i like dog", "he like cat", "they like animal"] # 训练数据、微型语料

    word_list = " ".join(sentences).split()
    word_list = list(set(word_list)) # 去重得到词汇表
    word_dict = {w: i for i, w in enumerate(word_list)} # 将每个词映射到一个唯一的索引
    number_dict = {i: w for i, w in enumerate(word_list)} # 将索引映射回词
    n_class = len(word_dict) # 词汇表的大小
    '''
    步骤三：实例化模型、损失函数与优化器
    '''
    model = NNLM() # 创建模型实例

    criterion = nn.CrossEntropyLoss() # 定义损失函数，交叉熵损失适用于多分类问题，内部自带了把原始得分转为概率的机制。
    optimizer = optim.Adam(model.parameters(), lr=0.001) # 定义优化器，使用Adam优化算法，学习率为0.001
    '''
    步骤四：准备喂给模型的 Tensor 数据
    '''
    input_batch, target_batch = make_batch() # 准备训练数据
    input_batch = torch.LongTensor(input_batch) # 将输入数据转换为LongTensor，适用于索引
    target_batch = torch.LongTensor(target_batch) # 将目标数据转换为LongTensor,适用于分类标签
    '''
    步骤五：训练模型
     - 迭代训练数据多次 epoch 每次迭代都要清空之前的梯度，计算模型输出，计算损失，反向传播计算梯度，并更新模型参数。
     - 每1000轮打印一次损失 ，以监控训练过程。
     - 训练完成后，使用模型进行预测，并将预测结果转换回词汇表中的词进行展示。
     - 预测结果是一个索引 需要通过number_dict将索引转换回对应的词。
     - 最后打印输入的上下文词和预测的下一个词，验证模型的学习效果。
    '''
    for epoch in range(5000):
        optimizer.zero_grad() # 清空之前的梯度
        output = model(input_batch) # 前向传播计算模型输出

        # 输出：[batch_size, n_class] target_batch: [batch_size] 交叉熵损失函数会自动将输出和目标进行对比，计算损失\
        loss = criterion(output, target_batch) # 计算损失
        if (epoch + 1) % 1000 == 0: # 每1000轮打印一次损失
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        
        loss.backward() # 反向传播计算梯度

        optimizer.step() # 更新模型参数
    # 预测
    predict = model(input_batch).data.max(1, keepdim=True)[1] # 获取预测结果，max(1)表示在类别维度上取最大值，keepdim=True保持维度不变，[1]表示取索引

    # Test
    print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()]) # 打印输入的上下文词和预测的下一个词