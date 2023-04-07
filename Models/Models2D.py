from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation, MaxPool2D, Concatenate


def conv_block(input, num_filters, batch_normalization=True):
    x = Conv2D(num_filters, 3, padding="same")(input)
    if batch_normalization:
        x = BatchNormalization()(x)   #Not in the original network.
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    if batch_normalization:
        x = BatchNormalization()(x)  #Not in the original network
    x = Activation("relu")(x)

    return x

#Encoder block: Conv block followed by maxpooling


def encoder_block(input, num_filters, batch_normalization):
    x = conv_block(input, num_filters, batch_normalization)
    p = MaxPool2D((2, 2))(x)
    return x, p

#Decoder block
#skip features gets input from encoder for concatenation

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

#Build Unet using the blocks
def build_unet2D(input_shape, first_layer_num_features, batch_normalization):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, first_layer_num_features, batch_normalization)
    s2, p2 = encoder_block(p1, first_layer_num_features*2, batch_normalization)
    s3, p3 = encoder_block(p2, first_layer_num_features*4, batch_normalization)
    s4, p4 = encoder_block(p3, first_layer_num_features*8, batch_normalization)

    b1 = conv_block(p4, first_layer_num_features*16, batch_normalization) #Bridge

    d1 = decoder_block(b1, s4, first_layer_num_features*8)
    d2 = decoder_block(d1, s3, first_layer_num_features*4)
    d3 = decoder_block(d2, s2, first_layer_num_features*2)
    d4 = decoder_block(d3, s1, first_layer_num_features)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)  #Binary (can be multiclass)

    model = Model(inputs, outputs, name="U-Net")
    return model
