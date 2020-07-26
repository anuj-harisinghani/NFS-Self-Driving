# resnet.py
# Residual Network 50 (ResNet50)

from keras import layers, models
from keras.applications import ResNet50

def ResNet(width, height):
    resnet = ResNet50(include_top = False, weights = 'imagenet', input_shape = (width, height, 3))
    model = models.Sequential()
    model.add(resnet)
    
    # model.add(layers.Flatten())
    model.add(layers.GlobalAveragePooling2D(data_format = None))
    model.add(layers.Dense(32, activation = 'sigmoid'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(3, activation = 'softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    
    print(model.summary())
    return model

ResNet(64, 48)