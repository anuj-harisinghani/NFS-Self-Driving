# balance_data.py

# Balance the data recorded. Most of the data is of output "straight" (approx 80% of it).
# We need to take equal amount of left, straight, right data.
# This script shuffles the complete dataset and then chooses n amount of samples.
# 'n' is the number of samples in the output that has the least frequency.

import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2
import time 

TRAINING_VERSION = 2
FILE_PART = 12

file_name = 'training_data_{}_{}.npy'.format(TRAINING_VERSION, FILE_PART)

# train_data = np.load(file_name, allow_pickle = True)
# train_data = np.load('training_data_2_balanced_400k.npy', allow_pickle = True)
train_data = np.load('training_data_2_400k_samples.npy', allow_pickle = True)
print("Length of dataset: " + str(len(train_data)))
#print("Sample data point:")

# check data
df = pd.DataFrame(train_data)
print(df.head())
print()
print(Counter(df[1].apply(str)))

'''
print()
print(train_data[0][0])
print()
print(train_data[0][1])
print()
print("[\n", train_data[0][0], ",", train_data[0][1], "\n]")

# show data - for debugging
for data in train_data:
    img = data[0]
    choice = data[1]
    cv2.imshow('test', img)
    print(choice)
    time.sleep(0.06)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
'''

#balance data
lefts = []
rights = []
forwards = []

shuffle(train_data)

for data in train_data:
    img = data[0]
    choice = data[1]

    if choice == [1,0,0]:
        lefts.append([img,choice])
    elif choice == [0,1,0]:
        forwards.append([img,choice])
    elif choice == [0,0,1]:
        rights.append([img,choice])
    else:
        print('no matches')


forwards = forwards[:len(lefts)][:len(rights)]
lefts = lefts[:len(forwards)]
rights = rights[:len(forwards)]

final_data = forwards + lefts + rights

shuffle(final_data)
print(len(final_data))

np.save('training_data_2_balanced_400k_test.npy', final_data)

train_data1 = np.load('training_data_2_balanced_400k_test.npy', allow_pickle = True)
# print(len(train_data1))
df1 = pd.DataFrame(train_data1)
print(Counter(df1[1].apply(str)))
