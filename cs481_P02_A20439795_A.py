# This coding exercise is done by 
#   - Jiecheng Mei
#   - Khiem Do
import json
import sys
import pandas as pd
import os
import time
from functools import reduce


# delete some character and split the sentence to words
def split_count_words(sentence, tag):
    massive_words = []
    chars_to_remove = [',', '.', '-', '!', '\"', ':', ')', '(']
    for char in chars_to_remove:
        sentence = sentence.replace(char, '')
    sentence = sentence.split()
    sentence = [element.lower() for element in sentence]
    for each in sentence:
        massive_words.append((str(each), int(tag)))
    return massive_words


# count the tag from massive dataset to store dataset
def count_words(dataset):
    store_data = {}
    total_label_counts = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}  # Initialize label counts
    
    for x in dataset:
        word, tag = x[0], x[1]
        tag_str = str(tag)  # Ensure tag is string for key consistency
        if word not in store_data:
            store_data[word] = {"total": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
        store_data[word]["total"] += 1
        store_data[word][tag_str] += 1
        total_label_counts[tag_str] += 1  # Update total label counts
    
    return store_data, total_label_counts


def write_to_local(dataset, file_name, dir_results):
    full_path = os.path.join(dir_results, file_name)
    with open(full_path, 'w') as file:
        json.dump(dataset, file, indent=2)


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
    data_train = df[:int(len(df) * t_size_f)]
    print(f'Length of the train dataset size: {len(data_train)}')
    _train_start = time.time()
    train_data_simplify = data_train[["Score", "Summary", "Text"]]
    train_dataset = pd.DataFrame(train_data_simplify)
    massive_train_dataset = []
    for i in range(1, len(train_dataset)):
        row = train_dataset.iloc[i]
        tag = row.iloc[0]
        summary = row.iloc[1]
        text = row.iloc[2]
        sentence = f'{summary} {text}'
        massive_train_dataset.extend(split_count_words(sentence, tag))
    print('[TRAIN PART] split words finished... Beginning count words')

    train_dataset = count_words(massive_train_dataset)
    write_to_local(train_dataset, f'{t_size}.json', dir_path)
    _train_enclape = time.time() - _train_start
    print(f"total train time take: {_train_enclape}")
    return train_dataset


def pre_process_test_data():
    dir_path = './dataset/test'
    Reviews_dataset = pd.read_csv('dataset/Reviews.csv')
    # remove useless column
    Reviews_dataset = Reviews_dataset[['Score', 'Summary', 'Text']]
    # get last 20% of data
    test_dataset = Reviews_dataset[int(len(Reviews_dataset)*0.8):]
    print(f'Length of the test dataset size: {len(test_dataset)}')
    _test_start = time.time()
    test_dataset = pd.DataFrame(test_dataset)
    massive_test_dataset = []
    for i in range(len(test_dataset)):
        row = test_dataset.iloc[i]
        tag = row.iloc[0]
        summary = row.iloc[1]
        text = row.iloc[2]
        sentence = f"{summary} {text}"
        massive_test_dataset.extend(split_count_words(sentence, tag))
    print("[TEST PART] split words finished... Beginning count words")

    test_dataset = count_words(massive_test_dataset)
    write_to_local(test_dataset, f'test.json', dir_path)
    _test_enclaps = time.time() - _test_start
    print(f'Total test time take: {_test_enclaps}')
    return test_dataset


# # Calculate the size of the dictionary
# def calculate_vocab_size(store_data):
#     return len(store_data)


# # P(label)
# def probability_of_label(label, total_label_counts):
#     total_counts = sum(total_label_counts.values())
#     return total_label_counts[label] / total_counts if total_counts > 0 else 0


# # P(word | label)
# def probability_of_word_given_label(word, label, store_data, total_label_counts, vocab_size):
#     word_data = store_data.get(word, {"total": 0, label: 0})
#     word_label_count = word_data[label] + 1  # Laplace Smoothing
#     total_words_for_label = sum(item[label] for item in store_data.values()) + vocab_size
#     return word_label_count / total_words_for_label


# def classify_sentence(sentence, store_data, total_label_counts, vocab_size):
#     words = split_count_words(sentence, '')  # Assumes preprocessing to match training
#     label_probs = {}
#     for label in total_label_counts:
#         label_prob = probability_of_label(label, total_label_counts)
#         word_probs = [probability_of_word_given_label(word[0], label, store_data, total_label_counts, vocab_size) for word in words]
#         label_probs[label] = label_prob * reduce(lambda x, y: x * y, word_probs, 1)
    
#     best_label = max(label_probs, key=label_probs.get)
#     return best_label


if __name__ == '__main__':
    # Global Variables
    tagged_word = {}
    train_size = 80
    _file_name = ''

    # This part handling parameter pass by command
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

    # Detect if the last 20% data on local, if not create one, else use it
    json_files_test = []
    _dir_path = './dataset/test'
    for filename in os.listdir(_dir_path):
        if filename.endswith('.json'):
            if filename not in json_files_test:
                json_files_test.append(filename)
    if "test.json" not in json_files_test:
        test_data = pre_process_test_data()
    else:
        print('Existing TEST dataset detected...')
        test_data = load_from_local('test.json', _dir_path)

    # Detect if the ##% train data on local, if not create it, else use it
    json_files_train = []
    _dir_path = './dataset/train'
    for filename in os.listdir(_dir_path):
        if filename.endswith('.json'):
            if filename not in json_files_train:
                json_files_train.append(filename)

    if f"{train_size}.json" not in json_files_train:
        train_data = pre_process_train_data(train_size)
    else:
        print('Existing TRAIN dataset detected...')
        train_data = load_from_local(f'{train_size}.json', _dir_path)

    print("[=TEST=] See this message means counter is working properly")

    # This part placehold for confusion matrix

    # This part placehold for Sentence with naive bayes classifier
    # P(label|S) = P(label)*P(word1|label)*P(word2|label)...

