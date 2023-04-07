import tensorflow as tf
from tensorflow.keras import backend as K
def dice_coef(y_true, y_pred, smooth=1e-7):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true, axis=-1) + K.sum(y_pred, axis=-1)
    return K.mean((2. * intersection + smooth) / (union + smooth))
def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)