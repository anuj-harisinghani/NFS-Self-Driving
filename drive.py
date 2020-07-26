# drive.py

# Real-time driving in NFS Most Wanted
# Open NFS MW in dxwnd to force it in 640x480 resolution. Follow the steps mentioned here https://nitrotech.info/force-window-mode-games-dxwnd/ .
# Open free roam mode and set the car on the road. The model was trained on the dashboard camera and so it will drive using the same. (2nd Camera setting.)

import numpy as np
from keras.models import load_model
import cv2
from grabscreen import grab_screen
from directkeys import PressKey, ReleaseKey, W, A, D
import time
from getkeys import key_check

from ac_net import ACNet
# from resnet import ResNet
# from dense_net import DenseNet

WIDTH = 64
HEIGHT = 48

VERSION = 4
EPOCHS = 50
SAMPLES = 400

MODEL = 'acnet'
MODEL_NAME = '{}-v{}-{}-epochs-{}k-samples.model'.format(MODEL, VERSION, EPOCHS, SAMPLES)

model = ACNet(WIDTH, HEIGHT)
# model = ResNet(WIDTH, HEIGHT)
# model = DenseNet(WIDTH, HEIGHT)

model = load_model(MODEL_NAME)

t_time = 0.05

# defining possible movements
def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    
def left():
    PressKey(W)
    PressKey(A)
    ReleaseKey(D)
    time.sleep(t_time)
    ReleaseKey(A)

def right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    time.sleep(t_time)
    ReleaseKey(D)
    
def stop():
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(W)
    
def main():
    print("Loaded model: " + MODEL_NAME)
    
    # 4 second countdown
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
    
    last_time = time.time()
    time_per_frame = 0
    iterations = 0

    paused = False
    while(True):
        if not paused:
            screen = grab_screen(region=(0, 0, 640, 480))
            screen = cv2.cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (64, 48))
            
            cv2.namedWindow('input screen')        
            cv2.moveWindow('input screen', 800,30)
            cv2.imshow('input screen', screen)
            
            # taking game frame as input and predicting a one-hot array at that exact frame
            img = np.array(screen).reshape(-1,64,48,1)                            # acnet
            # x2 = np.array(screen)                                               # resnet and densenet
            # img = np.repeat(x2[..., np.newaxis], 3, -1).reshape(-1, 64, 48, 3)  # resnet and densenet
            pred = model.predict(np.array(img))
            y = list(np.around(pred[0]))
            print(y)
            
            if y == [0, 1, 0]:
                straight()
            elif y == [0, 0, 1]:
                right()
            elif y == [1, 0, 0]:
                left()
        
        # press G to pause the script
        keys = key_check()
        if 'G' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)
                
        iterations+=1
        time_per_frame += float(format(time.time() - last_time))
        last_time = time.time()
                             
        # focus opencv window and press Q to quit
        if (cv2.waitKey(25) & 0xFF == ord('q')):
            stop()
            cv2.destroyAllWindows()
            print('Average FPS:', iterations/time_per_frame)
            break
        
main()