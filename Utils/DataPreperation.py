import os
import tifffile
import numpy as np
import random
from skimage import transform

X = []
y_MG = []
y_EA = []
names = []
path = '../Data/raw data/'
address_list = []
for root, dirs, files in os.walk(path, topdown=False):
    if "US" in root:
        # print(root)  # to show progress and debug
        if any(["final" in f for f in files]):  # This is to make sure that all raw data is segmented by EA and MG.
            # As a result of running the code for the first time, it was stated that 066, 068 band 087 don't have maks
            for name in files:
                imgAddress = os.path.join(root, name)
                if "initial combined" in name:
                    img = tifffile.imread(imgAddress)
                    X.append(img)
                    names.append(name)
                elif not any(["initial combined" in f for f in files]) and "initial" in name:
                    img = tifffile.imread(imgAddress)
                    X.append(img)
                    names.append(name)
                # Extracting masks
                if ("MG" in name) and ("US" in name):
                    img = tifffile.imread(imgAddress)
                    y_MG.append(img)
                    names.append(name)
                elif ("EA" in name) and ("US" in name):
                    img = tifffile.imread(imgAddress)
                    y_EA.append(img)
                    names.append(name)
        address_list.append(root)
        # print('------------\n')
# Now list X contains all the training pictures with tiff format, next we will save 3D tiff files as 2D images to training set
# Y_MG and Y_EA represent the masks created by two summer students


#----------------------------- Training Set -------------------------------

tiff_num = len(X)
test_num = 5
X_train= []
y_train_EA = []
y_train_MG = []
img_size=128
for sample in range(tiff_num-test_num):
    raw_X = np.array(X[sample])/255                                     #normalizing by deviding by 255
    raw_y_MG = np.array(y_MG[sample] != 0)
    raw_y_EA = np.array(y_EA[sample] != 0)
    for i in range(raw_X.shape[0]):
        for j in range(0, 12):
            rot_X = transform.rotate(raw_X[i], angle=j*30, resize=False)
            rot_y_MG = transform.rotate(raw_y_MG[i], angle=j*30, resize=False)
            rot_y_EA = transform.rotate(raw_y_EA[i], angle=j*30, resize=False)
            for k in range(2):
                a = random.randint(0, 5)
                b = random.randint(rot_X.shape[0]-5, rot_X.shape[0])
                c = random.randint(0, 5)
                d = random.randint(rot_X.shape[1]-10, rot_X.shape[1])
                cropped_X = transform.resize(rot_X[a:b, c:d], (img_size,img_size)).astype(np.float16)
                cropped_y_MG = transform.resize(rot_y_MG[a:b, c:d], (img_size,img_size)).astype(np.uint8)
                cropped_y_EA = transform.resize(rot_y_EA[a:b, c:d], (img_size,img_size)).astype(np.uint8)
                X_train.append(cropped_X)
                y_train_MG.append(cropped_y_MG)
                y_train_EA.append(cropped_y_EA)
    print("sample", sample, "augmented")

X_train=np.array(X_train)
y_train_EA=np.array(y_train_EA)
y_train_MG=np.array(y_train_MG)
y_train_combined=np.logical_or(y_train_EA, y_train_MG).astype(np.uint8)
#------------------------------ Test Set -------------------------------

tiff_num = len(X)
test_num = 5
X_test= []
y_test_EA = []
y_test_MG = []
for sample in range(tiff_num-test_num, tiff_num):
    raw_X = np.array(X[sample])
    raw_y_MG = np.array(y_MG[sample] != 0)
    raw_y_EA = np.array(y_EA[sample] != 0)
    for i in range(raw_X.shape[0]):
        for j in range(0, 12):
            rot_X = transform.rotate(raw_X[i], angle=j*30, resize=False)
            rot_y_MG = transform.rotate(raw_y_MG[i], angle=j*30, resize=False)
            rot_y_EA = transform.rotate(raw_y_EA[i], angle=j*30, resize=False)
            for k in range(2):
                a = random.randint(0, 5)
                b = random.randint(rot_X.shape[0]-5, rot_X.shape[0])
                c = random.randint(0, 5)
                d = random.randint(rot_X.shape[1]-10, rot_X.shape[1])
                cropped_X = transform.resize(rot_X[a:b, c:d], (img_size,img_size)).astype(np.float16)
                cropped_y_MG = transform.resize(rot_y_MG[a:b, c:d], (img_size,img_size)).astype(np.uint8)
                cropped_y_EA = transform.resize(rot_y_EA[a:b, c:d], (img_size,img_size)).astype(np.uint8)
                X_test.append(cropped_X)
                y_test_MG.append(cropped_y_MG)
                y_test_EA.append(cropped_y_EA)
    print("sample", sample, "augmented")

X_test=np.array(X_test)
y_test_EA=np.array(y_test_EA)
y_test_MG=np.array(y_test_MG)
y_test_combined=np.logical_or(y_test_EA, y_test_MG).astype(np.uint8)


np.save('../Data/preprocessed data/X_train.npy', X_train)
np.save('../Data/preprocessed data/X_test.npy', X_test)
np.save('../Data/preprocessed data/y_train_combined.npy', y_train_combined)
np.save('../Data/preprocessed data/y_test_combined.npy', y_test_combined)

