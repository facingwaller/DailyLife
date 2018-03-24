import t_cnn_dict

Dict_path = 'D:\\tensorflow\CNN_SHUXIN\T_data.txt'
train_data_path = 'D:\\tensorflow\CNN_SHUXIN\Atr.txt'

'''
train_data = open(train_data_path, 'r', encoding='utf-8').readlines()

Dic = t_cnn_dict.Dict(Dict_path)
#print(Dic)
label = []

for i in train_data:
    label.append(Dic[i])

#print(label)
#print(label[0])
#print(label[77])
#print(label[218])
#print(label[1434])
'''


def get_label(Dict_path, train_data_path):
    label =[]
    train_data = open(train_data_path, 'r', encoding='utf-8').readlines()
    #print(train_data)
    Dic = t_cnn_dict.Dict(Dict_path)                   # 调用函数构造字典

    # 遍历原始数据，添加标签到标签数组
    for i in train_data:
        i = i.strip()
        #print(i)
        label.append(Dic[i])
    return label


'''
if __name__ == '__main__':
    print(get_label(Dict_path, train_data_path))
'''






