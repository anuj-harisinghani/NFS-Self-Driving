# train.py
# train your models here

from ac_net import ACNet
import numpy as np
import matplotlib.pyplot as plt

# load balanced dataset
data = np.load('training_data_2_balanced_400k.npy', allow_pickle = True)
# data = np.load('3_training_data_2_balanced_400k.npy', allow_pickle = True)

WIDTH = 64
HEIGHT = 48

VERSION = 3
EPOCHS = 50
SAMPLES = 400
MODEL = 'acnet'

MODEL_NAME = '{}-v{}-{}-epochs-{}k-samples.model'.format(MODEL, VERSION, EPOCHS, SAMPLES)

# initialize a new model with the given width and height

model = ACNet(WIDTH,HEIGHT)

# train-test split = 75-5-20 (TRAIN, VAL, TEST) %
# 400k samples balanced data- FROM 87939, TRAIN = 65954, VAL = 4397, TEST = 17588 (65954 + 4397 = 70351)
# 400k samples balanced data- FROM 92820, TRAIN = 69615, VAL = 4641, TEST = 18564 (69615 + 4641 = 74556)
train = data[:-18564]                                    # removed testing data from total 
val = train[-4641:]                                      # removed val from train data
test = data[-18564:]                                     # reserved test data from total

# train data
X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
Y = [i[1] for i in train]

# val data
val_x = np.array([i[0] for i in val]).reshape(-1,WIDTH,HEIGHT,1)
val_y = [i[1] for i in val]

# test data
test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
test_y = [i[1] for i in test]

# train model
history = model.fit(x = np.array(X), 
                    y = np.array(Y), 
                    batch_size = 512, 
                    validation_data = (np.array(val_x), np.array(val_y)), 
                    epochs=EPOCHS) 

# plot accuracy over epochs
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('Model Accuracy (Categorical Accuracy) - {}'.format(MODEL_NAME))
plt.ylabel('Categorical Accuracy')
plt.xlabel('Epoch')
plt.legend(['training', 'validation'], loc='upper left')
plt.show()

# plot loss over epochs
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss (Categorical Crossentropy)- {}'.format(MODEL_NAME))
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['training', 'validation'], loc='upper left')
plt.show()


model.save(MODEL_NAME)
