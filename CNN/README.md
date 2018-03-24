data_helpe.py/
train.py/
text_cnn.py/
eval.py
以上四个分别为text-cnn的原始代码，其中train，text_cnn为改动过的

t_cnn_num.py       为对数据提取属，去掉重复属性，统计所有非重复属性
t_cnn_dict.py      对属性建立字典
t_cnn_label.py     建立标签
t_wiki_process.py  对wiki处理，建立字-字向量字典，并且匹配中文

Atr.txt:所有属性
T_dat.txt :去掉重复后的属性
只用到这两个文件
