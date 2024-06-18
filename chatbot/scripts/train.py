import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import pickle

# Function to load and clean stories from a CSV file
def load_and_clean_stories(csv_file):
    df = pd.read_csv(csv_file)
    stories = df['story'].tolist()
    cleaned_stories = [clean_text(story) for story in stories]
    return cleaned_stories

# Function to clean text
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text

# Load and clean the stories
csv_file = "../data/stories.csv"
cleaned_stories = load_and_clean_stories(csv_file)
all_stories = ' '.join(cleaned_stories)

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([all_stories])
total_words = len(tokenizer.word_index) + 1

# Create input sequences
input_sequences = []
token_list = tokenizer.texts_to_sequences([all_stories])[0]
for i in range(1, len(token_list)):
    n_gram_sequence = token_list[:i+1]
    input_sequences.append(n_gram_sequence)

# Pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

X, y = input_sequences[:,:-1], input_sequences[:,-1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Define the model
model = Sequential()
model.add(Embedding(total_words, 64, input_length=max_sequence_len-1))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=100, verbose=1)

# Save the model and tokenizer
model.save("../models/story_generator.h5")
with open("../models/tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle)
