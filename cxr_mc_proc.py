import pandas as pd
import numpy as np
import cv2
import random

df = pd.read_csv('/home/nyh/cxr_mc/Data_Entry_2017.csv')

image_index = df['Image Index']
finding_labels = df['Finding Labels']
pd.DataFrame(pd.value_counts(finding_labels)).to_csv('/home/nyh/cxr_mc/stats.csv')
disease_list = pd.DataFrame(pd.value_counts([a for a in finding_labels if '|' not in a]))
disease_list.to_csv('/home/nyh/cxr_mc/unique_stats.csv')

input_dir = '/home/nyh/cxr_mc/images/'
output_dir = '/home/nyh/cxr_mc/scaled/'
save_dir = '/home/nyh/cxr_mc/saved/'

def proc(disease_name):
    image_list = list(image_index[finding_labels == disease_name])
    image_data = np.zeros((len(image_list), 224, 224, 1), 'uint8')
    for i, fn in enumerate(image_list):
        if i % 100 == 0:
            print(disease_name, '{}/{}'.format(i, len(image_list)), fn)
        image = cv2.imread(input_dir + fn)
        image = cv2.resize(image, (224, 224))
        image_data[i, :, :, 0] = image[:, :, 0]


    print('saving ...')
    np.save(save_dir + disease_name + '.npy', image_data)


# from concurrent.futures import ThreadPoolExecutor
# with ThreadPoolExecutor(10) as e:
#     for dname in disease_list.index:
#         e.submit(proc, dname)

import keras.utils
class_n = 10
train_x = []
train_y = []
test_x = []
test_y = []
val_x = []
val_y = []
for i, dname in enumerate(disease_list.index[:class_n]):
    print(dname)
    print('reading npy ...')
    image_data = np.load(save_dir + dname + '.npy')
    n_image = image_data.shape[0]
    n_test = (n_image//8)
    n_val = 50
    n_train = 3000
    test_x.append(image_data[-n_test:])
    test_y.append(keras.utils.to_categorical([i]*n_test, class_n))
    val_x.append(image_data[-(n_test+n_val):-n_test])
    val_y.append(keras.utils.to_categorical([i]*n_val, class_n))
    train_x.append(image_data[
        np.random.choice(range(0, n_image-(n_test+n_val)),n_train)
    ])
    train_y.append(keras.utils.to_categorical([i]*n_train, class_n))
print('concatenating...')
np.save(save_dir + 'train_x.npy', np.concatenate(train_x))
np.save(save_dir + 'train_y.npy', np.concatenate(train_y))
np.save(save_dir + 'val_x.npy', np.concatenate(val_x))
np.save(save_dir + 'val_y.npy', np.concatenate(val_y))
np.save(save_dir + 'test_x.npy', np.concatenate(test_x))
np.save(save_dir + 'test_y.npy', np.concatenate(test_y))
