# This coding exercise is done by 
#   - Jiecheng Mei
#   - Khiem Do
import sys
import pandas as pd
import os


def pre_process_data(t_size: float):
    df = pd.read_csv('dataset/Reviews.csv')
    # print(df.iloc[row, column]) this return the location value from the array
    # print(f'row in df:{len(df)}') this return the total row from the array
    data_test = df[int(len(df) * 0.8):]
    data_train = df[1:int(len(df) * train_size)]
    test_data_simplify = data_test.iloc[:, [6, 8, 9]]
    train_data_simplify = data_train.iloc[:, [6, 8, 9]]
    return test_data_simplify, train_data_simplify


if __name__ == '__main__':
    # handles the input from the parameter
    tagged_word = {}
    train_size = 80

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

    # handles if the dataset is existed
    csv_files = []
    directory = './dataset/results'
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filename = filename.split('.' and '_')
            if int(filename[0]) not in csv_files:
                csv_files.append(int(filename[0]))
    if train_size in csv_files:
        print('Existing data file occur, automatically using pre trained data')

    else:
        train_size = train_size / 100
        test_data, train_data = pre_process_data(train_size)


    # This part placehold for store count table to CSV

    # This part placehold for last 20% test

    # This part placehold for confusion matrix

    # This part placehold for Sentence with naive bayes classifier
    # P(label|S) = P(label)*P(word1|label)*P(word2|label)...

    while True:
        userInput = input("Enter your sentence:\nSentence S:")
        # placehold for naive bayes classifier
