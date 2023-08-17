
import numpy as np
from keras.optimizers import Adam
from Utils.lossFunction import dice_coef, dice_loss
from Models.Models2D import build_unet2D
import matplotlib.pyplot as plt


#------------------------------ Model Building -------------------------------
X_train=np.load('Data/preprocessed data/X_train.npy')
X_test=np.load('Data/preprocessed data/X_test.npy')
y_train_combined=np.load('Data/preprocessed data/y_train_combined.npy')
y_train_combined = y_train_combined.astype('float32')
y_test_combined=np.load('Data/preprocessed data/y_test_combined.npy')
y_test_combined = y_test_combined.astype('float32')

img_size=X_train[1].shape[0]
input_shape = (img_size, img_size, 1) # 1 represents number of channels which is one for greyscale
model = build_unet2D(input_shape, 32, True)
model.compile(optimizer=Adam(learning_rate = 1e-4), loss=dice_loss, metrics=[dice_coef, 'mean_squared_error'])
model.summary()

history = model.fit(X_train, y_train_combined,
                    batch_size=64,
                    verbose=1,
                    epochs=40,
                    validation_split=0.05,
                    shuffle=False)

model.save('Models/Trained Models/128by128UNET32filters.hdf5')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['train', 'val'], loc='upper right')
plt.show()


plt.plot(history.history['dice_coef'])
plt.plot(history.history['val_dice_coef'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['train', 'val'], loc='lower right')
plt.show()

y_pred = model.predict(X_test)

# calculate dice coefficient and loss on test set
test_results = model.evaluate(X_test, y_test_combined, verbose=1)
print(test_results)
a=105
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 10))
# Display the first image in the first subplot
axes[0, 0].imshow(X_test[a], cmap='gray')
axes[0, 0].set_title('specimen 1')

# Display the second image in the second subplot
axes[0, 1].imshow(X_test[a+100], cmap='gray')
axes[0, 1].set_title('specimen 2')

axes[0, 2].imshow(X_test[a+200], cmap='gray')
axes[0, 2].set_title('specimen 3' )

axes[0, 3].imshow(X_test[a+300], cmap='gray')
axes[0, 3].set_title('specimen 4')

axes[1, 0].imshow(y_test_combined[a], cmap='gray')
axes[1, 0].set_title('Ground Truth 1')

# Display the second image in the second subplot
axes[1, 1].imshow(y_test_combined[a+100], cmap='gray')
axes[1, 1].set_title('Ground Truth 2')

axes[1, 2].imshow(y_test_combined[a+200], cmap='gray')
axes[1, 2].set_title('Ground Truth 3')

axes[1, 3].imshow(y_test_combined[a+300], cmap='gray')
axes[1, 3].set_title('Ground Truth 4')

axes[2, 0].imshow(y_pred[a], cmap='gray')
axes[2, 0].set_title('Model Segmentation 1')

# Display the second image in the second subplot
axes[2, 1].imshow(y_pred[a+100], cmap='gray')
axes[2, 1].set_title('Model Segmentation 2')

axes[2, 2].imshow(y_pred[a+200], cmap='gray')
axes[2, 2].set_title('Model Segmentation 3')

axes[2, 3].imshow(y_pred[a+300], cmap='gray')
axes[2, 3].set_title('Model Segmentation 4')
fig.suptitle('Breast Cancer Segmentation Results \n(second row represents the segmentation done by human, third row shows the segmentation carried out by the model)', fontsize=16)
plt.show()