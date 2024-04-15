import tensorflow as tf
import numpy as np
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)
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
loss_fn = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
model = NeuralNetwork()
epochs = 10000
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        predictions = model(X)
        loss = loss_fn(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: Loss: {loss.numpy()}")
predictions = model(X)
print("Predictions:")
print(predictions.numpy())
