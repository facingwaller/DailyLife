Atr_num = open("Atr_num.txt", 'w', encoding='utf-8')
Atr = open('Atr.txt', 'w', encoding='utf-8')
Atr_One = open('Atr_One.txt', 'w', encoding='utf-8')
T_data = open('T_data.txt', 'w', encoding='utf-8')

path = 'D:\\tensorflow\q.rdf.ms.re.v1.filter.txt'
# out_data =[]

# 把数据中的属性提取出来存入属性列表，并统计重复的属性构建一个字典
with open(path, 'r', encoding='utf-8-sig') as f:
    atrribute = []
    Re = {}
    Re_Not = {}

    # 统计所有属性加到列表
    for sentence in f:
        file = sentence.split('\t')
        atrribute.append(file[3])
    for i in atrribute:   #输出到文本保存
        print(i, file=Atr)
        print(i)
    print(len(atrribute))

    # 统计重复属性,并计数加到字典
    for i in atrribute:
        if atrribute.count(i) > 1:  #计数，大于1的就添加进去
            Re[i] = atrribute.count(i)
        else:
            Re[i] = 1  # 单独统计只出现一次的属性，加入原字典
            Re_Not[i] = 1  # 单独统计只出现一次的属性，加到另外一个字典，只是用来区分用的
    for key, value in Re.items():
        print("{}:{}".format(key, value), file=Atr_num)
    for key, value in Re_Not.items():
        print("{}:{}".format(key, value), file=Atr_One)
    print(len(Re))
    print(len(Re_Not))

# 把字典的键值即属性加到列表中
out_data = tuple(Re.keys())
for i in out_data:
    print(i, file=T_data)
print(len(out_data))

