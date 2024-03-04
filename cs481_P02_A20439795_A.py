# This coding exercise is done by 
#   - Jiecheng Mei
#   - Khiem Do
# ====================================================
# Hint from Professor
# once have the classifier done, save it into CSV file
# binary decision model, hashtable will work, but our model might not work
# ====================================================
# For the dataset, I put in into a folder first then place into it.
# You NEED to download the file yourself, it is too big so we can't upload to GitHub
# the directory is:
#       Original dataset: dataset/Reviews.csv
#       After clean-up: dataset/Reviews_clean.csv
#       After classifier: dataset/Reviews_classifier.csv
# ====================================================

# ++++++++++++++++++
# Important note:
# test_data := the last 20% data will be assign in test_data after read file
# train_data := the {tran_size}% train data will be assign in train_data
# ++++++++++++++++++

import sys
import pandas as pd

if __name__ == '__main__':
    # handle the input from the parameter
    # this will restrict the input between 20 and 80 and if 'yes' receive 80 instead.
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

    print(f'Training set size: {train_size}%')
    train_size = train_size / 100

    # This part placehold for read in data
    df = pd.read_csv('dataset/Reviews.csv')
    # print(df.iloc[7, 8]) this return the location value from the array
    # print(f'row in df:{len(df)}') this return the total row from the array
    test_data = df[int(len(df)*0.8):]
    train_data = df[:int(len(df)*train_size)]

    # This part placehold for build classifier

    # This part placehold for store classifier to CSV

    # This part placehold for last 20% test

    # This part placehold for confusion matrix

    # This part placehold for Sentence with naive bayes classifier


