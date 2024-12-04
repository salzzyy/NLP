# ANN vs. RNN

| **Feature**           | **ANN (Artificial Neural Networks)**        | **RNN (Recurrent Neural Networks)**           |
|-----------------------|---------------------------------------------|-----------------------------------------------|
| **Architecture**       | Feedforward neural network.                | Recurrent network with loops to allow connections back to previous layers. |
| **Flow of Data**       | Data flows in one direction (input → output). | Data flows in loops to retain historical context. |
| **Memory**            | No memory of previous inputs.              | Maintains a memory of prior inputs using hidden states. |
| **Task Type**         | Suitable for static data (e.g., image classification). | Suitable for sequential or time-dependent data (e.g., text, time series). |
| **Applications**      | Image classification, regression tasks.    | Language modeling, time series prediction.    |
| **Hidden States**     | No hidden states involved.                 | Maintains hidden states that are updated as the sequence is processed. |
| **Vanishing Gradient** | Occurs with deep networks.                 | More prone to vanishing gradients due to sequential backpropagation. |
| **Long-term Memory**  | Not designed to retain context.            | Struggles with long-term dependencies (solved by LSTMs and GRUs). |
| **Example Use Cases** | Predicting house prices from static features. | Predicting the next word in a sentence.       |
| **Temporal Awareness** | Not suitable for temporal patterns.        | Captures dependencies across sequences of data. |
| **Complexity**        | Simpler to implement and train.            | More complex due to sequential processing and hidden states. |
| **Data Type**         | Works well with static, tabular, or image data. | Specialized for sequential data like text, speech, or time series. |
| **Training Time**     | Relatively faster due to simpler structure. | Slower because of sequence-based backpropagation (BPTT). |
| **Examples of Models**| Multilayer Perceptron (MLP), Convolutional Neural Networks (CNNs). | Vanilla RNN, LSTM, GRU. |
| **Dependencies**      | Independent processing of input data.      | Contextual processing, considering relationships between inputs. |
| **Mathematical Model**| \( y = f(Wx + b) \) (output depends on current input only). | \( h_t = f(Wx_t + Uh_{t-1} + b) \) (output depends on current and previous inputs). |

