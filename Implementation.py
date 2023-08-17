import os
import tifffile
import numpy as np
from skimage import transform
from keras.models import load_model
from keras.losses import binary_crossentropy
from Utils.lossFunction import dice_coef, dice_loss
import keras.utils


keras.utils.get_custom_objects().update({'dice_coef': dice_coef, 'dice_loss': dice_loss})

# Load preprocessed data
def preprocess_image(img_path, img_size=128):
    img = tifffile.imread(img_path)
    img_list = []
    for i in range(img.shape[0]):
        img_slice = img[i]
        resized_slice = transform.resize(img_slice, (img_size, img_size)).astype(np.float16)
        img_list.append(resized_slice)
    return np.array(img_list)

# Load the trained model
def load_trained_model(model_path):
    model = load_model(model_path, custom_objects={'dice_coef': dice_coef, 'binary_crossentropy': binary_crossentropy})
    return model

# Load ground truth segmentations if available
def load_ground_truth(path):
    y_MG = None
    y_EA = None
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if "MG" in name:
                y_MG = tifffile.imread(os.path.join(root, name))
            elif "EA" in name:
                y_EA = tifffile.imread(os.path.join(root, name))
    return y_MG, y_EA

# Load the model
model_path = 'Models/Trained Models/128by128UNET32filters.hdf5'
model = load_trained_model(model_path)

# Input path
input_dir = 'Data/test_image_directory/'  # Replace this with the path containing the test image and segmentations

tiff_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.tif')]

# Loop through the TIFF files and process each one
for tiff_file in tiff_files:
    # Construct the full path to the TIFF image
    tiff_path = os.path.join(input_dir, tiff_file)

    # Preprocess the image
    X_test = preprocess_image(tiff_path)

    # Normalize input data to range [0, 1]
    # X_test = X_test.astype('float32') / 255.0

    # Print information for debugging
    print("Processing:", tiff_file)
    print("X_test shape:", X_test.shape)

    # Perform segmentation
    y_pred = model.predict(X_test)

    # Create a directory for the segmented images
    output_dir = 'Results/segmented_images'
    image_name = os.path.splitext(tiff_file)[0]
    output_dir = os.path.join(output_dir, image_name)
    os.makedirs(output_dir, exist_ok=True)

    # Print information for debugging
    print("Output directory:", output_dir)

    # Save the segmented images in the new directory
    for i in range(X_test.shape[0]):
        original_slice = (X_test[i] * 255).astype(np.uint8)
        segmented_slice = (y_pred[i, :, :, 0]*255).astype(np.uint8)
        combined_image = np.hstack((original_slice, segmented_slice))

        # Print information for debugging
        print("Saving slice:", i)

        output_path = os.path.join(output_dir, f'slice_{i}.png')
        tifffile.imwrite(output_path, combined_image)

print("Segmentation completed.")