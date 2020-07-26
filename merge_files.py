# merge_files.py
# after creating separate small dataset files, merge them together into one big dataset file

import numpy as np

# data from file_name2 will be appended to file_name
file_name = 'training_data_2_400k_samples.npy'
file_name2 = 'training_data_2_16.npy'

def main():
    data1 = list(np.load(file_name2, allow_pickle = True))
    training_data = list(np.load(file_name, allow_pickle = True))
    for data in data1:
        img = data[0]
        output = data[1]
        print(output)
        training_data.append([img,output])
    
    np.save(file_name,training_data)
main()
