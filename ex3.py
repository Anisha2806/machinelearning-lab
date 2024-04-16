# app.py
import streamlit as st
import tensorflow as tf
import numpy as np

# Define the neural network model
class NeuralNetwork(tf.keras.Model):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.input_layer = tf.keras.layers.Dense(2, activation='sigmoid')
        self.hidden_layer = tf.keras.layers.Dense(2, activation='sigmoid')
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.hidden_layer(x)
        return self.output_layer(x)

# Function to train the model
def train_model(X, y, epochs=10000):
    model = NeuralNetwork()
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            predictions = model(X)
            loss = loss_fn(y, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return model

# Function to make predictions
def make_predictions(model, X):
    predictions = model(X)
    return predictions.numpy()

# Main function to run the Streamlit app
def main():
    st.title('Simple Neural Network with Streamlit')

    # Sample input data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    # Train the model
    model = train_model(X, y)

    # Display predictions
    st.subheader('Predictions')
    predictions = make_predictions(model, X)
    
    # Output predictions
    for i, prediction in enumerate(predictions):
        st.write(f"Input: {X[i]}, Predicted Output: {prediction[0]}")

if __name__ == '__main__':
    main()
