# ac_net.py
# Autonomous Cars Network (ACNet)
# Convolutional Neural Network that takes 64x48 grayscale image and outputs either left, right or straight

from keras import layers, models

def ACNet(width, height):
    model = models.Sequential()
    model.add(layers.Conv2D(8, kernel_size = (5, 5), activation='relu', strides = (2, 2), input_shape=(width, height, 1)))
    model.add(layers.MaxPooling2D(pool_size = (2,2)))
    
    model.add(layers.Conv2D(36, kernel_size = (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size = (2,2)))
    
    model.add(layers.Conv2D(128, kernel_size = (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size = (2,2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='sigmoid'))
    model.add(layers.Dense(32, activation = 'sigmoid'))
    model.add(layers.Dense(3, activation = 'softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    
    print(model.summary())
    return model

ACNet(64, 48)
    
