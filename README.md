### Project Summary: Sentiment Analysis using LSTM and Binary Classification

This project focuses on **Sentiment Analysis** using a deep learning approach with **LSTM (Long Short-Term Memory)** layers in a Sequential model for binary classification of text data. The goal is to classify text inputs (such as product reviews) as either **Positive** or **Negative** sentiments.

#### Key Components:
1. **Data Preprocessing**:
   - Text data is tokenized using the `Tokenizer` from Keras, and each sentence is converted to a sequence of integers.
   - The sequences are then padded to ensure equal length for input into the neural network (max length set to 50).

2. **Model Architecture**:
   - The model begins with an **Embedding layer** that converts input words into dense vectors of fixed size.
   - Two **LSTM layers** (with 128 and 64 units) are stacked to capture sequential patterns in the text data.
   - A **Dense layer** (64 units) is added, followed by a **Dropout layer** to prevent overfitting.
   - The final output is a **single neuron** with a sigmoid activation function, outputting a probability between 0 and 1 (for binary classification).

3. **Model Training**:
   - The model is compiled using the **binary cross-entropy loss function** and **Adam optimizer**.
   - The model is trained to distinguish between positive and negative sentiments in the dataset.
   - The final model is saved as `model.h5`, and the tokenizer is saved as `tokenizer.pickle` for later use.

4. **Model Prediction**:
   - For predicting sentiment, input text is tokenized, converted to padded sequences, and fed into the model.
   - The output is a probability: values above 0.5 are classified as **Positive**, and below 0.5 as **Negative**.
   - Confidence scores are provided for each prediction, indicating the model's certainty.

5. **Test and User Input**:
   - A list of test cases can be passed through the model for evaluation.
   - User input functionality is implemented, where users can input custom sentences, and the model will return the predicted sentiment along with a confidence score.

#### Model Summary:
- **Input**: Sequences of up to 50 tokens (after padding).
- **Layers**:
  1. **Embedding Layer**: Converts tokenized words into 100-dimensional vectors.
  2. **LSTM Layers**: Two stacked LSTM layers capture temporal dependencies in the sequence.
  3. **Dense Layer**: 64 units for further processing.
  4. **Dropout Layer**: Helps prevent overfitting.
  5. **Output Layer**: A single unit with a sigmoid activation for binary classification (Positive/Negative).
- **Output**: A probability score for sentiment classification.

#### Tools and Libraries Used:
- **TensorFlow / Keras**: For building and training the LSTM model.
- **Pandas & Matplotlib**: For handling data and visualizing predictions.
- **NumPy**: For numerical computations.
- **Jupyter Notebook / Google Colab**: For developing and running the project.

#### Code Functions:
- **SentimentAnalysis(text)**: Tokenizes and processes the input text, runs it through the model, and outputs the sentiment and confidence.
- **test_on_data(test_data)**: Evaluates a batch of test cases and prints the corresponding predictions.
- **predict_user_input()**: Allows users to input custom text and get real-time sentiment predictions.

#### Applications:
- **Customer Reviews**: Classifying reviews as positive or negative can help businesses better understand customer feedback.
- **Social Media Monitoring**: Track sentiment in social media posts.
- **Chatbots**: Improve chatbot interactions by understanding the user's emotional state.

### Conclusion:
This project demonstrates the power of deep learning techniques, specifically LSTM, in understanding and classifying sentiments from text data. With the ability to analyze text as either positive or negative, this model can be applied in various domains where understanding user sentiment is essential.
