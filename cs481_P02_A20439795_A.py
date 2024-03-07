# This coding exercise is done by 
#   - Jiecheng Mei
#   - Khiem Do
import json
import sys
import pandas as pd
import os


# 删除干扰物并且标记单词
def split_words(sentence, tag):
    massive_words = []
    chars_to_remove = [',', '.', '-', '!', '\"']
    for char in chars_to_remove:
        sentence = sentence.replace(char, '')
    sentence = sentence.split()
    sentence = [element.lower() for element in sentence]
    for each in sentence:
        massive_words.append((each, tag))
    return massive_words


# 数数， 返回只有一个label的数值，假如需要别的label，需要从main那改变
def count_words(dataset, tag):
    store_data = {}
    for i in dataset:
        # create a key if dictionary doesn't exist a key
        if i[0] not in store_data:
            store_data[i[0]] = [tag, 0, 0, 0]
        # total count
        store_data[i[0]][1] += 1
        # count with tag or count without tag
        if i[1] == tag:
            store_data[i[0]][2] += 1
        else:
            store_data[i[0]][3] += 1
    return store_data


def write_to_local(dataset, file_name, dir_results):
    full_path = os.path.join(dir_results, file_name)
    with open(full_path, 'w') as file:
        json.dump(dataset, file)


def load_from_local(file_name, dir_results):
    full_path = os.path.join(dir_results, file_name)
    with open(full_path, 'r') as file:
        data_dict = json.load(file)
    return data_dict


# 这个部分读去数据并返回逐行数据本身，并且只获取score，summary，text
def pre_process_train_data(t_size: int):
    dir_path = './dataset/train'
    df = pd.read_csv('dataset/Reviews.csv')
    t_size_f = t_size / 100
    # print(f'[=TEST=] train_size in method = {t_size_f}')
    data_train = df[1:int(len(df) * t_size_f)]
    train_data_simplify = data_train.iloc[:, [6, 8, 9]]
    massive_words_dataset = []
    for i in range(1, len(train_data_simplify)):
        row = train_data_simplify.iloc[i]
        tag = row.iloc[0]
        summary = row.iloc[1]
        text = row.iloc[2]
        sentence = f'{summary} {text}'
        massive_words_dataset = split_words(sentence, tag)
    train_dataset_1 = count_words(massive_words_dataset, 1)
    train_dataset_2 = count_words(massive_words_dataset, 2)
    train_dataset_3 = count_words(massive_words_dataset, 3)
    train_dataset_4 = count_words(massive_words_dataset, 4)
    train_dataset_5 = count_words(massive_words_dataset, 5)
    write_to_local(train_dataset_1, f'{t_size}_1.json', dir_path)
    write_to_local(train_dataset_2, f'{t_size}_2.json', dir_path)
    write_to_local(train_dataset_3, f'{t_size}_3.json', dir_path)
    write_to_local(train_dataset_4, f'{t_size}_4.json', dir_path)
    write_to_local(train_dataset_5, f'{t_size}_5.json', dir_path)
    return train_dataset_1, train_dataset_2, train_dataset_3, train_dataset_4, train_dataset_5


def pre_process_test_data():
    dir_path = './dataset/test'
    df = pd.read_csv('dataset/Reviews.csv')
    # this +1 because the column name is existing on the row 0 in df
    # to eliminate it, must be +1
    data_test = df[int(len(df) * 0.8) + 1:]
    test_data_simplify = data_test.iloc[:, [6, 8, 9]]
    for i in range(1, len(test_data_simplify)):
        row = test_data_simplify.iloc[i]
        tag = row.iloc[0]
        summary = row.iloc[1]
        text = row.iloc[2]
        sentence = f"{summary} {text}"
        massive_words_dataset = split_words(sentence, tag)
    test_dataset_1 = count_words(massive_words_dataset, 1)
    test_dataset_2 = count_words(massive_words_dataset, 2)
    test_dataset_3 = count_words(massive_words_dataset, 3)
    test_dataset_4 = count_words(massive_words_dataset, 4)
    test_dataset_5 = count_words(massive_words_dataset, 5)
    write_to_local(test_dataset_1, f'test_1.json', dir_path)
    write_to_local(test_dataset_2, f'test_2.json', dir_path)
    write_to_local(test_dataset_3, f'test_3.json', dir_path)
    write_to_local(test_dataset_4, f'test_4.json', dir_path)
    write_to_local(test_dataset_5, f'test_5.json', dir_path)
    return test_dataset_1, test_dataset_2, test_dataset_3, test_dataset_4, test_dataset_5


if __name__ == '__main__':
    # 全局变量
    tagged_word = {}
    train_size = 80
    _file_name = ''

    # 这个部分处理从terminal发来的指令
    if len(sys.argv) == 2:
        try:
            arg_val = int(sys.argv[1])
            if 20 <= arg_val <= 80:
                train_size = arg_val
            else:
                raise ValueError
        except ValueError:
            pass
    print(f'Mei Jiecheng A20439795, Khiem Do A20483713 solution:\n'
          f'Training set size = {train_size}%')

    # 检测最后20%的数据是否在本地
    json_files_test = []
    _dir_path = './dataset/test'
    for filename in os.listdir(_dir_path):
        if filename.endswith('.json'):
            filename = filename.split('.')
            if filename[0] not in json_files_test:
                json_files_test.append(filename[0])

    if len(json_files_test) != 5:
        test_data1, test_data2, test_data3, test_data4, test_data5 = pre_process_test_data()
    else:
        print('Existing TEST dataset detected...')
        test_data1 = load_from_local('test_1.json', _dir_path)
        test_data2 = load_from_local('test_2.json', _dir_path)
        test_data3 = load_from_local('test_3.json', _dir_path)
        test_data4 = load_from_local('test_4.json', _dir_path)
        test_data5 = load_from_local('test_5.json', _dir_path)

    # train的文件格式是 ##_#.JSON ##代表train_size,#代表label
    json_files_train = []
    _dir_path = './dataset/train'
    for filename in os.listdir(_dir_path):
        if filename.endswith('.json'):
            filename = filename.split('.' and '_')
            if int(filename[0]) not in json_files_train:
                json_files_train.append(int(filename[0]))

    if train_size in json_files_train:
        print('Existing TRAIN dataset detected...')
        train_data1 = load_from_local(f'{train_size}_1.json', _dir_path)
        train_data2 = load_from_local(f'{train_size}_2.json', _dir_path)
        train_data3 = load_from_local(f'{train_size}_3.json', _dir_path)
        train_data4 = load_from_local(f'{train_size}_4.json', _dir_path)
        train_data5 = load_from_local(f'{train_size}_5.json', _dir_path)
    else:
        train_data1, train_data2, train_data3, train_data4, train_data5 = pre_process_train_data(train_size)
    print("[=TEST=] See this message means counter is working properly")
    # This part placehold for confusion matrix

    # This part placehold for Sentence with naive bayes classifier
    # P(label|S) = P(label)*P(word1|label)*P(word2|label)...

