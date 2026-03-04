"""
Handwritten Digit Recognition - CNN Model Training Script
This script trains a Convolutional Neural Network (CNN) on the MNIST dataset
to recognize handwritten digits (0-9).
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

def load_and_preprocess_data():
    """
    Load the MNIST dataset and preprocess it for training.
    - Normalizes pixel values to range [0, 1]
    - Reshapes images to include channel dimension
    - One-hot encodes the labels
    """
    print("Loading MNIST dataset...")
    
    # Load the MNIST dataset (60,000 training images, 10,000 test images)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    print(f"Training samples: {x_train.shape[0]}")
    print(f"Test samples: {x_test.shape[0]}")
    print(f"Image shape: {x_train.shape[1]}x{x_train.shape[2]}")
    
    # Normalize pixel values from [0, 255] to [0, 1]
    # This helps the neural network train faster and more effectively
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape images to add channel dimension (28, 28) -> (28, 28, 1)
    # CNNs expect input in format (height, width, channels)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    # One-hot encode the labels (0-9 -> 10-dimensional vectors)
    # Example: 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    print("Data preprocessing complete!")
    return (x_train, y_train), (x_test, y_test)


def build_cnn_model():
    """
    Build a Convolutional Neural Network for digit recognition.
    Architecture:
    - 2 Convolutional layers with MaxPooling
    - Dropout for regularization
    - Dense layers for classification
    """
    print("Building CNN model...")
    
    model = models.Sequential([
        # First Convolutional Block
        # 32 filters of size 3x3, ReLU activation
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        # Max pooling reduces spatial dimensions by half
        layers.MaxPooling2D((2, 2)),
        
        # Second Convolutional Block
        # 64 filters of size 3x3, more filters to capture complex patterns
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third Convolutional Block
        # 64 filters for deeper feature extraction
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Flatten the 3D output to 1D for dense layers
        layers.Flatten(),
        
        # Dense hidden layer with 64 neurons
        layers.Dense(64, activation='relu'),
        
        # Dropout layer to prevent overfitting (randomly drops 50% of connections)
        layers.Dropout(0.5),
        
        # Output layer with 10 neurons (one for each digit 0-9)
        # Softmax activation outputs probability distribution
        layers.Dense(10, activation='softmax')
    ])
    
    # Compile the model
    # Adam optimizer: adaptive learning rate optimization
    # Categorical crossentropy: loss function for multi-class classification
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model architecture summary
    model.summary()
    
    return model


def train_model(model, x_train, y_train, x_test, y_test):
    """
    Train the CNN model on the MNIST dataset.
    - Uses batch size of 128
    - Trains for 10 epochs
    - Validates on test set during training
    """
    print("\nStarting model training...")
    
    # Train the model
    history = model.fit(
        x_train, y_train,
        batch_size=128,           # Number of samples per gradient update
        epochs=10,                # Number of times to iterate over the entire dataset
        validation_data=(x_test, y_test),  # Data for validation after each epoch
        verbose=1                 # Show training progress
    )
    
    return history


def evaluate_and_save_model(model, x_test, y_test):
    """
    Evaluate the trained model on test data and save it.
    """
    # Evaluate on test set
    print("\nEvaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save the trained model in H5 format
    model_path = 'digit_model.h5'
    model.save(model_path)
    print(f"\nModel saved as '{model_path}'")
    
    return test_accuracy


def main():
    """
    Main function to orchestrate the training pipeline.
    """
    print("=" * 60)
    print("Handwritten Digit Recognition - CNN Training")
    print("=" * 60)
    
    # Step 1: Load and preprocess data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Step 2: Build the CNN model
    model = build_cnn_model()
    
    # Step 3: Train the model
    history = train_model(model, x_train, y_train, x_test, y_test)
    
    # Step 4: Evaluate and save the model
    accuracy = evaluate_and_save_model(model, x_test, y_test)
    
    print("\n" + "=" * 60)
    if accuracy >= 0.97:
        print(f"SUCCESS! Model achieved {accuracy * 100:.2f}% accuracy (target: 97-99%)")
    else:
        print(f"Model achieved {accuracy * 100:.2f}% accuracy. Consider tuning hyperparameters.")
    print("=" * 60)


if __name__ == "__main__":
    main()
