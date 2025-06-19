import cv2
import keras
import numpy as np
import os
import sys
import tensorflow as tf


from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)
    print("running main ... Training completed! Now evaluating...")

    # Evaluate neural network performance
    # model.evaluate(x_test,  y_test, verbose=2)
    model.evaluate(x_test,  y_test)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"running main ... Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    images = []
    labels = []

    # Iteate across all the categories, e.g. 1, 2, 3, etc.
    for category in range(NUM_CATEGORIES):
        
        # Combine the data_dir (passed in) and the category (the current index number in 0 to NUM_CATEGORIES)
        category_dir = os.path.join(data_dir, str(category))

        # Ensure that this path (e.g. this folder) actuall exists
        if not os.path.exists(category_dir):
            continue 

        # Open all the files in that category, creating a 
        for file in os.listdir(category_dir):

            # Construct the path to this image
            image_path = os.path.join(category_dir, file)

            try:

                # Read the image using cv2 (per the instructions)
                image = cv2.imread(image_path)

                # Ensure the image was loaded correctly
                if image is None:
                    print(f"running load_data ... error loading image_path: {image_path}, skipping image")
                    continue
                
                # Resize the image 
                image_resized = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                #print(f"running load_data ... image_resized.shape is: {image_resized.shape}")

                # Normalize image
                image_normalized = image_resized.astype('float32') / 255.0

                # Add the image to the images list and the label to the labels list
                images.append(image_normalized)
                labels.append(category)

            except Exception as e:
                print(f"running load_data ... error processing image_path: {image_path}, hit error: {e}")
                raise ValueError           

    print(f"running load_data ... function completed, returning populated lists images[] and labels[]")
    return (images, labels)

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    
    Explaination of what each layer does:

    CONVOLUTIONAL LAYERS (Feature Detection):
    Conv layer 1 (32 filters): Edge detection → detects basic edges, lines, curves
    Pool layer 1: Reduces spatial dimensions while keeping important edge info
    Conv layer 2 (32 filters): Shape detection → combines edges into shapes, textures, patterns
    Pool layer 2: Further dimensionality reduction, focuses on most important shapes

    Flatten: Converts 2D feature maps to 1D for dense layers

    DENSE LAYERS (Feature Combination & Decision Making):
    Hidden layer 1 (128 units): Basic shape combinations → "circular red thing", "triangular outline", "rectangular blue area"
    Hidden layer 2 (64 units): Complex concepts → "stop sign vs speed limit sign", "warning vs regulatory vs informational"
    Output layer (43 units): Final classification → specific traffic sign identification
    """

    conv_layers = build_conv_pooling_layers(
        num_layers_conv=2,
        nodes_per_conv_layer=32,
        kernel_size=(3, 3),
        activation="relu",
        pool_size=(2, 2)
    ) 

    hidden_layers = build_hidden_layers(
        num_layers_hidden=1,
        first_hidden_layer_nodes=128,
        subsequent_hidden_layer_nodes_decrease=0.50,
        hidden_layer_activation_algo="relu",
        first_hidden_layer_dropout=0.5,
        subsequent_hidden_layer_dropout_decrease=0.20
    )

    # Create a convolutional neural network
    model = keras.models.Sequential([

        # Seperate input layer and convolutional layer, per suggestion in logs
        keras.layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

        # This unpacks the list of layers returned by the build_conv_pooling_layers function
        *conv_layers,

        # Flatten units
        keras.layers.Flatten(),
       
        # Add hidden layers (unpack the layers produced by build_hidden_layers above)
        # use of dropout prevents overfitting
        *hidden_layers,

        # Add an output layer with output units for all 10 digits
        # 10 categroies to represent each digit 0-9
        # Uses softmax activation function (turns output into a probability)
        # Use "sigmoid" if binary classification or "softmax" if using multiple categories
        keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # Compile the model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(f"running get_model ... function completed, returning model")
    return model


def build_conv_pooling_layers(
    num_layers_conv: int = 2,
    nodes_per_conv_layer: int = 32,
    kernel_size: tuple = (3, 3),
    activation: str = "relu",
    pool_size: tuple = (2, 2)        
):
    
    # Create structure to hold all the layers
    conv_layers = []
    for layer_num in range(num_layers_conv):
        
        conv_layer = keras.layers.Conv2D(
            nodes_per_conv_layer, kernel_size, activation=activation
        )
        conv_layers.append(conv_layer)
        
        # Add pooling layer  
        pool_layer = keras.layers.MaxPooling2D(pool_size=pool_size)
        conv_layers.append(pool_layer)

        print(f"running build_conv_pooling_layers ... completed layer_num: {layer_num} with \n"
              f"nodes_per_conv_layer: {int(nodes_per_conv_layer)}, \n"
              f"kernal_size: {kernel_size}, \n"
              f"activation: {activation}, \n"
              f"pool_size: {pool_size}")

    print(f"running build_conv_pooling_layers ... finished building conv_layers: {conv_layers}")
    return conv_layers


def build_hidden_layers(
    num_layers_hidden: int = 2,
    first_hidden_layer_nodes: int = 128,
    subsequent_hidden_layer_nodes_decrease: float = 0.50,
    hidden_layer_activation_algo: str = "relu",
    first_hidden_layer_dropout: float = 0.50,
    subsequent_hidden_layer_dropout_decrease: float = 0.10
):
    hidden_layers = []

    current_nodes = first_hidden_layer_nodes
    current_dropout = first_hidden_layer_dropout

    for layer_num in range(num_layers_hidden):
        # Add dense layer
        hidden_layers.append(
            keras.layers.Dense(int(current_nodes), activation=hidden_layer_activation_algo)
        )

        # Add dropout layer
        hidden_layers.append(
            keras.layers.Dropout(current_dropout)
        )

        # Calculate next layer's parameters
        current_nodes = max(8, current_nodes * subsequent_hidden_layer_nodes_decrease)
        current_dropout = max(0.1, current_dropout - subsequent_hidden_layer_dropout_decrease)

        print(f"running build_hidden_layers ... completed layer_num: {layer_num}: : {int(current_nodes)} nodes, {current_dropout:.2f} dropout")
    
    print(f"running build_hidden_layers ... finished building hiddden_layers: {hidden_layers}")
    return hidden_layers


if __name__ == "__main__":
    main()
