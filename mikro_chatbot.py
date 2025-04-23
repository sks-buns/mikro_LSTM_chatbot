import nltk
import numpy as np
import json
import pickle
import random
from nltk.stem import WordNetLemmatizer
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
sentences, labels = [], []
classes = []
ignore_chars = ['?', '!', ',', '.']

# Load dataset
try:
    file_path = r"C:\Users\LENOVO\OneDrive\Desktop\Desktop\Mikro ChatBot\Mikro_LSTM_Dataset.json"
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print("Please ensure the Mikro_LSTM_Dataset.json file is in the correct location")
    exit(1)

# Process dataset
for intent in data['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words = [lemmatizer.lemmatize(w.lower()) for w in word_list if w not in ignore_chars]
        sentences.append(" ".join(words))
        labels.append(intent['tag'])
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

classes = sorted(set(classes))

# Tokenize sentences
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
word_index = tokenizer.word_index

# Pad sequences
max_length = max(len(seq) for seq in sequences)
x_train = pad_sequences(sequences, maxlen=max_length, padding='post')

# Convert labels to numerical form
label_dict = {label: index for index, label in enumerate(classes)}
y_train = np.array([label_dict[label] for label in labels])

# Save tokenizer and classes
pickle.dump(tokenizer, open('tokenizer.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Build LSTM model
model = Sequential([
    Embedding(input_dim=len(word_index) + 1, output_dim=128, mask_zero=True),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(64)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(classes), activation='softmax')
])

# Compile model
model.compile(loss='sparse_categorical_crossentropy', 
             optimizer=Adam(learning_rate=0.001), 
             metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, epochs=100, batch_size=8, verbose=1)

# Save the model
model.save('chatbot_model.keras')

# Chatbot response function
def chatbot_response(text):
    words = [lemmatizer.lemmatize(w.lower()) for w in nltk.word_tokenize(text)]
    seq = tokenizer.texts_to_sequences([" ".join(words)])
    padded_seq = pad_sequences(seq, maxlen=max_length, padding='post')
    
    prediction = model.predict(padded_seq, verbose=0)[0]
    tag = classes[np.argmax(prediction)]
    
    for intent in data['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm not sure about that. Can you ask something else?"

# Main chat loop
if __name__ == "__main__":
    print("Chatbot is ready! Type 'exit' to stop.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Thank you for using Airbus chatbot tool. Goodbye!")
            break
        print("Bot: ", chatbot_response(user_input))