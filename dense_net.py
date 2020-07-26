# dense_net.py

from keras import layers, models
from keras.applications import DenseNet121

def DenseNet(width, height):
    densenet = DenseNet121(include_top = False, weights = 'imagenet', input_shape = (width, height, 3))
    model = models.Sequential()
    model.add(densenet)
    
    # model.add(layers.Flatten())
    model.add(layers.GlobalAveragePooling2D(data_format = None))
    model.add(layers.Dense(128, activation='tanh'))
    model.add(layers.Dense(32, activation = 'tanh'))
    model.add(layers.Dense(3, activation = 'softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    
    print(model.summary())
    return model

DenseNet(64, 48)