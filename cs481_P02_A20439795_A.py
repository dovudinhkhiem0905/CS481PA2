# This coding exercise is done by 
#   - Jiecheng Mei
#   - Khiem Do
# ====================================================
# Hint from Professor
# once have the classifier done, save it into CSV file
# binary decision model, hashtable will work, but our model might not work
# ====================================================
import sys
from sys import argv

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

    # finished the first printing, conver train size back to float number for future calculation
    train_size = train_size / 100

    # This part placehold for read in data

    # This part placehold for build classifier

    # This part placehold for store classifier to CSV

    # This part placehold for
