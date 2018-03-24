import numpy as np

data_path = 'D:\\tensorflow\CNN_SHUXIN\T_data.txt'
Dictionary = open('Dictionary.txt', 'w', encoding='utf-8')

'''
file = open(path, 'r', encoding='utf-8-sig').readlines()
#for i in file:
    #print(i)
# print(file[0])
size = len(file)
# print(size)

unit_mat = np.eye(size, dtype='int')
# unit_mat = list(unit_mat)
# print(unit_mat)
# print(unit_mat[1])

Dic = {}
# Dic = {'1':[1,0], '2':[0,1]}
# print(Dic)

count = 0
for i in file:
    # print(i)
    label = unit_mat[count]
    Dic[i] = label
    count += 1
print(Dic,file=Dictionary)
print(Dic)

out_data = tuple(Dic.keys())
print(out_data[1].strip('\n'))
# for i in out_data:
    # print(i)
print(out_data)
'''


# 构建属性标签字典
def Dict(data_path):
    Dic = {}
    count = 0
    file = open(data_path, 'r', encoding='utf-8-sig').readlines()
    size = len(file)
    unit_mat = np.eye(size, dtype='int')


    for i in file:
        i = i.strip()
        label = unit_mat[count]
        Dic[i] = label
        count += 1
    print(Dic, file=Dictionary)

    # out_data = tuple(Dic.keys())
    # out_data[1].strip('\n')
    return Dic

'''
if __name__ == '__main__':
    dic = Dict(data_path)
    print(dic)
    for i in dic:
        label = dic[i]
        print(label)
'''