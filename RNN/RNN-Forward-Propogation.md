# RNN Forward Propagation with Time

### Overview
Recurrent Neural Networks (RNNs) are designed to process sequential data, where the output at each time step depends not only on the current input but also on the hidden state from the previous time step. This makes RNNs well-suited for tasks such as language modeling, time series forecasting, and speech recognition.

### Mathematical Formulation

At each time step \( t \), the forward propagation in an RNN is as follows:

1. **Hidden State Calculation**:
\[
h_t = f(Wx_t + Uh_{t-1} + b)
\]
- \( h_t \): Hidden state at time step \( t \).
- \( x_t \): Input at time step \( t \).
- \( h_{t-1} \): Hidden state from the previous time step.
- \( W \): Weight matrix for the input.
- \( U \): Recurrent weight matrix for the hidden state.
- \( b \): Bias term.
- \( f \): Activation function (typically Tanh or ReLU).

2. **Output Calculation** (optional):
\[
y_t = g(Vh_t + c)
\]
- \( y_t \): Output at time step \( t \).
- \( V \): Weight matrix for the hidden state.
- \( c \): Bias term for the output.
- \( g \): Activation function for the output layer (e.g., softmax for classification).

### Example

Let’s say we have a simple sequence where we predict the next number in a series.

Input sequence: [1, 2, 3, 4]

1. **At \( t = 1 \)**:
   - \( x_1 = 1 \)
   - \( h_1 = f(Wx_1 + b) \)
   - \( y_1 = g(Vh_1 + c) \)

2. **At \( t = 2 \)**:
   - \( x_2 = 2 \)
   - \( h_2 = f(Wx_2 + Uh_1 + b) \)
   - \( y_2 = g(Vh_2 + c) \)

This process continues for each time step, with the hidden state \( h_t \) depending on both the current input and the previous hidden state.

### Python Code Example (using Keras/TensorFlow)

```python
import numpy as np
import tensorflow as tf

# Example data
X = np.array([1, 2, 3, 4]).reshape(-1, 1)  # Input sequence (reshaped to (time_steps, features))

# Build a simple RNN model
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(10, activation='tanh', input_shape=(1, 1), return_sequences=True),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Reshape X to (batch_size, time_steps, features)
X = X.reshape((1, 4, 1))

# Fit the model
model.fit(X, X, epochs=100, verbose=0)

# Predict
predictions = model.predict(X)
print(predictions)






### Key Concepts:
- **Hidden State Update**: Each hidden state depends on both the input and the previous hidden state.
- **Sequential Nature**: The model retains memory of previous inputs, making it suitable for time-dependent tasks.
  

