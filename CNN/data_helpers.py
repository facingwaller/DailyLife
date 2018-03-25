import numpy as np
import re
import itertools
from collections import Counter
import t_cnn_dict
import t_cnn_label



def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    # 正则替换,相应的替换规则需要去弄懂，这个函数应该是用来对数据进行预处理
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()   # 移除头尾字符串（如果里面没写东西，默认空格等空白符），并改为小写


# 读取数据和labels
def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r",encoding="utf-8").readlines())  # readlines()一次性读所有行，readline()一次读一行
    positive_examples = [s.strip() for s in positive_examples]  #  s.strip(rm) 当rm为空时，默认删除空白符（包括'\n', '\r',  '\t',  ' ')
    negative_examples = list(open(negative_data_file, "r",encoding="utf-8").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]   # [XX for i in YY]这是一个链表推导式，［x*y for x in num1 for y in num2]嵌套for循环的感觉
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]  # "_"就是省略的，可丢弃的意思？,positive_labels输出的就是一个[[0, 1], [0, 1], ...，[0, 1], [0, 1], [0, 1]]的矩阵，同反例
    print(positive_labels)
    negative_labels = [[1, 0] for _ in negative_examples]
    print(negative_labels)
    y = np.concatenate([positive_labels, negative_labels], 0)  # np.concatenate（）数组的拼接，0为行拼接，直接在原有数组上加框，1为列拼接，把数加到框中，返回的是[[0,1],[1,0]]
    print("1=================================")
    print(x_text)
    print("2=================================")
    # print(y)
    print("3=================================")
    print(len(x_text))
    print(len(y))
    print("4=================================")
    return [x_text, y]              # 这里的的return[]返回的是一个的单词文字的数组(列表)？

'''
  """
    下面这个是返回的格式
    [
        [
            "hello good ",
            " bad bad "
        ]
        [
            [0,1],
            [1,0]
        ]    
    ]
  """
'''



# 读取数据和labels-QA
def load_data_and_labels_QA(train_data_path):

    # Load data from files
    dict_data_path = '../data/T_data.txt'
    train_data = open(train_data_path, 'r', encoding='utf-8').readlines()
    input_data = [s.strip() for s in train_data]  #  s.strip(rm) 当rm为空时，默认删除字符串首尾空白符（包括'\n', '\r',  '\t',  ' ')
    x_text = input_data

    # Generate labels
    label = t_cnn_label.get_label(dict_data_path, train_data_path)
    y = np.array(label)
    print("1=================================")
    print(x_text)
    print("2=================================")
    print(y)
    print(y.shape)
    print("3=================================")
    print(len(x_text))
    print(len(y))
    print("4=================================")

    return [x_text, y]


# 定义batch样本的迭代生成器
# num_epochs表示训练过程中，将被训练多少次
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1  # 看英文应该是每个epoch的batch数，但是个人感觉像取数据的波数，和num_epochs（训练的波数）不是很同
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size)) # np.random.permutation（）就是对数据重新乱序，并且返回一个副本
            shuffled_data = data[shuffle_indices]     # 这种类型的表达形式是参照上述的打乱过后的一维矩阵shuffle_indices的数值作为下标，调整原本矩阵data的数值顺序
        else:
            shuffled_data = data  # 即shuffle=False,则数据不洗牌，还是原来的数据
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size  # batch_num从0开始，取数据batch的起点
            end_index = min((batch_num + 1) * batch_size, data_size) #  取数据batch的终点，一般data_size大于batch，后面就不一样,可能取不到个数，所以取min
            yield shuffled_data[start_index:end_index]  # 返回取的那段batch数据,即在shuffled_data这个矩阵中，从start开始取，一共取   end-start  个数据（可超越上限，超越上限则取满）


#if __name__ == '__main__':
#    print(load_data_and_labels_QA('D:\\tensorflow\CNN-SHUXIN\Atr.txt'))
