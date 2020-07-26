# test_model.py
# test your trained models here

import numpy as np
from keras.models import load_model

# load dataset and split
data = np.load('training_data_2_balanced_400k.npy', allow_pickle = True)
train = data[:-18564]
test = data[-18564:]

# input data (x) is primarily a numpy array of size 64x48 with grayscale pixel values. 
X = np.array([i[0] for i in train]).reshape(-1,64,48,1)
x_test = np.array([i[0] for i in test]).reshape(-1,64,48,1)

# output data (y) is in the form of a one-hot array (example- [1, 0, 0])
# the position of the 1 in the array determines left, straight or right
# left - [1, 0, 0]; straight - [0, 1, 0]; right - [0, 0, 1]
Y = [i[1] for i in train]
y_test = [i[1] for i in test]

# model
VERSION = 3
EPOCHS = 50
SAMPLES = 400
MODEL = 'acnet'

MODEL_NAME = '{}-v{}-{}-epochs-{}k-samples.model'.format(MODEL, VERSION, EPOCHS, SAMPLES)
model = load_model(MODEL_NAME)

loss, acc = model.evaluate(np.array(x_test),  np.array(y_test), verbose=2)
print('Restored model {} \naccuracy on {} test samples: {:5.2f}%'.format(MODEL_NAME, len(test), 100*acc))

'''
# check predictions manually here
y_pred = []
for i in x_test:
    i = np.array(i).reshape(-1,64,48,1)
    pred = model.predict(np.array(i))
    y = list(np.around(pred[0]))
    y_pred.append(y)
print(y_pred)
'''