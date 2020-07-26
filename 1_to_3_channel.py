# 1_to_3_channel.py
# use this code to convert 1 channel dataset to 3 channels for networks that require 3 channel images as input
# ResNet50, DenseNet and many other CNNs require 3 channel input images
# this code repeats the 1 channel file 3 times 

import numpy as np

# data = np.load('3_training_data_2_balanced_400k.npy', allow_pickle = True)
data2 = np.load('training_data_2_balanced_400k.npy', allow_pickle = True)
data1 = []

for d in data2:
    img = np.repeat(d[0][..., np.newaxis], 3, -1)
    output = d[1]
    data1.append([img, output])
    
np.save('3_channel_data.npy', data1)


'''
x = data[0][0]
x2 = data2[0][0]
print("x shape = ", x.shape)
print("x2 shape = ", x2.shape)
y = x.reshape(-1, 64, 48, 3)
y2 = np.repeat(x2[..., np.newaxis], 3, -1)
print("y shape =", y.shape)
print("y2 shape = ", y2.shape)
y3 = y2.reshape(-1, 64, 48, 3)
print("y3 shape = ", y3.shape)
'''