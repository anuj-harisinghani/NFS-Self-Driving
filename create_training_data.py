# create_training_data.py
# captures training dataset and stores in .npy files

import numpy as np
from grabscreen import grab_screen
import cv2
import time
from getkeys import key_check
import os

# specify training data file to be created or accessed
TRAINING_VERSION = 2
FILE_PART = 16

file_name = 'training_data_{}_{}.npy'.format(TRAINING_VERSION, FILE_PART)

# append captured dataset to existing file, else create new file
if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    training_data = list(np.load(file_name, allow_pickle = True))
else:
    training_data = []

# record keys pressed
def keys_to_output(keys):
    output = [0,0,0]
    
    if 'A' in keys:
        output[0] = 1
    elif 'D' in keys:
        output[2] = 1
    else:
        output[1] = 1
    return output
    

def main():
    # 4 second countdown
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
    
    while(True):
        # capture screen
        screen = grab_screen(region=(0, 0, 640, 480))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen, (64, 48))
        
        # capture keys
        keys = key_check()
        output = keys_to_output(keys)
        training_data.append([screen,output])
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        
        # save file after every 1000 data points
        if len(training_data) % 1000 == 0:
            print(len(training_data))
            np.save(file_name,training_data)
            
main()
