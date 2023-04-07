# from keras import backend as K
# def dice_coef(y_true, y_pred):
#     y_true_f = K.cast(y_true, dtype='float32')
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
#     return (2. * intersection + K.epsilon()) / (union + K.epsilon())
#
# def dice_coef_loss(y_true, y_pred):
#     return -dice_coef(y_true, K.cast(y_pred, 'float32'))

import tensorflow as tf
from tensorflow.keras import backend as K

@tf.function
def dice_coef(y_true, y_pred, smooth=1e-7):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true, axis=-1) + K.sum(y_pred, axis=-1)
    return K.mean((2. * intersection + smooth) / (union + smooth))

@tf.function
def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)