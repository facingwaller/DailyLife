wikis_list = []
words_embedding = []
wiki_dir = {}
one_list =[]
all_list =[]


# 把wiki数据变为字典
def get_wiki_dic(wiki_data_path):
    wiki_num = []
    with open(wiki_data_path, 'r', encoding='utf-8') as wiki:
        for i in wiki:                    # 读出wiki vector中的每行
            i = i.strip()
            wikis_list.append(i)
        for word in wikis_list:           # 这个for循环 取出头部的字作为键值
            word = word.split(' ')
            number = word[1:]
            for i in number:              # 这个for循环 获取字之后的数字，加入到数字列表,作为值
                i = eval(i)
                wiki_num.append(i)
            wiki_dir[word[0]] = wiki_num  # 建立字典
            # print(word[0])
            # print(wiki_num)
            # print(wiki_dir)
            # print(len(wiki_dir))
            # print('\n')
            wiki_num = []
    return wiki_dir



# 在字典中匹配文字向量
def get_vector(word,wiki_dir):
    #with open(Atr_path, 'r', encoding='utf-8') as relation:
    #for word in relation.readlines():
    one_list = []
    word = word.strip()
    for one_word in word:
        list =wiki_dir[one_word]
        one_list.append(list)
    return one_list

'''
if __name__ == '__main__':
    print(get_vector("作者", get_wiki_dic('D:\\tensorflow\wiki.vector')))
'''