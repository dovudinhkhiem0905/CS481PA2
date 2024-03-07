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


def write_to_local(dataset, file_name):
    dir_results = './dataset/results'
    full_path = os.path.join(dir_results, file_name)
    with open(full_path, 'w') as file:
        json.dump(dataset, file)


def load_from_local(file_name):
    dir_results = './dataset/test'
    full_path = os.path.join(dir_results, file_name)
    with open(full_path, 'r') as file:
        data_dict = json.load(file)
    return data_dict


# 这个部分读去数据并返回逐行数据本身，并且只获取score，summary，text
def pre_process_train_data(t_size: float):
    df = pd.read_csv('dataset/Reviews.csv')
    data_train = df[1:int(len(df) * t_size)]
    train_data_simplify = data_train.iloc[:, [6, 8, 9]]
    massive_words_dataset = []

    for element in train_data_simplify:
        tag = element[0]
        sentence = str(element[1] + ' ' + element[2])
        massive_words_dataset = split_words(sentence, tag)

    train_dataset_1 = count_words(massive_words_dataset, 1)
    train_dataset_2 = count_words(massive_words_dataset, 2)
    train_dataset_3 = count_words(massive_words_dataset, 3)
    train_dataset_4 = count_words(massive_words_dataset, 4)
    train_dataset_5 = count_words(massive_words_dataset, 5)
    write_to_local(train_dataset_1, f'{t_size}_1.json')
    write_to_local(train_dataset_2, f'{t_size}_2.json')
    write_to_local(train_dataset_3, f'{t_size}_3.json')
    write_to_local(train_dataset_4, f'{t_size}_4.json')
    write_to_local(train_dataset_5, f'{t_size}_5.json')
    return train_dataset_1, train_dataset_2, train_dataset_3, train_dataset_4, train_dataset_5


def pre_process_test_data():
    df = pd.read_csv('dataset/Reviews.csv')
    # this +1 because the column name is existing on the row 0 in df
    # to eliminate it, must be +1
    data_test = df[int(len(df) * 0.8) + 1:]
    test_data_simplify = data_test.iloc[:, [6, 8, 9]]
    for element in test_data_simplify:
        tag = element[0]
        sentence = str(element[1] + ' ' + element[2])
        massive_words_dataset = split_words(sentence, tag)

    test_dataset_1 = count_words(massive_words_dataset, 1)
    test_dataset_2 = count_words(massive_words_dataset, 2)
    test_dataset_3 = count_words(massive_words_dataset, 3)
    test_dataset_4 = count_words(massive_words_dataset, 4)
    test_dataset_5 = count_words(massive_words_dataset, 5)
    write_to_local(test_dataset_1, f'test_1.json')
    write_to_local(test_dataset_2, f'test_2.json')
    write_to_local(test_dataset_3, f'test_3.json')
    write_to_local(test_dataset_4, f'test_4.json')
    write_to_local(test_dataset_5, f'test_5.json')
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
    _file_name = f'{train_size}_train.csv'

    # 检测最后20%的数据是否在本地
    for filename in os.listdir('./dataset'):
        if filename.endswith('.csv'):
            filename.split('.')
            if filename == 'Tests.csv':
                test_data = pd.read_csv('Tests.csv')
            else:
                test_data = pre_process_test_data()

    # 检测是否有已经做过的数据
    csv_files = []
    directory = './dataset/results'
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filename = filename.split('.' and '_')
            if int(filename[0]) not in csv_files:
                csv_files.append(int(filename[0]))
    if train_size in csv_files:
        print('Existing data file occur, automatically using pre trained data')
        # laod the existing {train_size}_train.csv
        # load the existing {train_size}_test.csv
    else:
        train_size = train_size / 100
        train_data = pre_process_train_data(train_size)

    # This part placehold for last 20% test

    # This part placehold for confusion matrix

    # This part placehold for Sentence with naive bayes classifier
    # P(label|S) = P(label)*P(word1|label)*P(word2|label)...

    while True:
        userInput = input("Enter your sentence:\nSentence S:")
        # placehold for naive bayes classifier
