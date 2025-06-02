#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Bidirectional, LSTM, Embedding
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model

print("tensorflow/libs ready")


# In[2]:


#DOWNLOAD STOP WORDS AND LEMMATIZER
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# In[3]:


file_path = r'C:\Users\tyler\OneDrive - SNHU\WGU\Neural Networks\Task 2\sentiment labelled sentences\yelp_labelled.txt'

texts = []
labels = []

with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        parts = line.rsplit('\t', 1) 
        if len(parts) == 2:
            text, label = parts
            text = text.strip().lower()
            texts.append(text)
            labels.append(int(label.strip()))

# Convert texts into a TensorFlow tensor
texts_tensor = tf.constant(texts)

# Remove non-ASCII characters (e.g., emojis, non-English characters)
cleaned_texts = tf.strings.regex_replace(texts_tensor, r'[^\x00-\x7F]+', '')  # Remove non-ASCII characters
cleaned_texts = tf.strings.regex_replace(cleaned_texts, r'[^\w\s]', '')  # Remove non-word characters (punctuation)

# Remove stopwords
def remove_stopwords(text):
    # Split tensor into individual words
    tokens = tf.strings.split(text)
    # Convert tokens to lowercase
    lower_tokens = tf.strings.lower(tokens)
    # Use TensorFlow to create a mask for non-stopwords
    mask = tf.reduce_any([tf.strings.regex_full_match(lower_tokens, stop_word) for stop_word in stop_words], axis=0)
    filtered_tokens = tf.boolean_mask(lower_tokens, ~mask)
    return tf.strings.join(filtered_tokens, separator=' ')

# Apply stopword removal
filtered_texts = [remove_stopwords(text) for text in cleaned_texts]

# Lemmatization - Use NLTK as TensorFlow doesn't provide a built-in lemmatizer
def lemmatize_text(text):
    words = text.numpy().decode('utf-8').split()  # Convert TensorFlow tensor to string and split
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(lemmatized_words)

lemmatized_texts = [lemmatize_text(text) for text in filtered_texts]

# Tokenizer: Tokenize lemmatized texts
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lemmatized_texts)

# Convert texts to sequences
sequences = tokenizer.texts_to_sequences(lemmatized_texts)

# Print sample sequences
print("\nSample sequences (stopwords removed, lemmatized):")
for i in range(5):
    print(f"Original Text: {texts[i]}")
    print(f"Cleaned Text (stopwords removed, lemmatized): {lemmatized_texts[i]}")
    print(f"Sequence: {sequences[i]}")
    print()

vocab_size = len(tokenizer.word_index)
print(vocab_size)


# In[4]:


#count positive and negative
label_counts = Counter(labels)
positive_count = label_counts[1]
negative_count = label_counts[0]

print(f"Positive labels (1): {positive_count}")
print(f"Negative labels (0): {negative_count}")


# In[5]:


#calculate sentence lengths
sentence_lengths = [len(sentence.split()) for sentence in texts]

#distribution of sentaence lengths
plt.hist(sentence_lengths, bins=50, color='blue', alpha=0.7)
plt.title('Distribution of Sentence Lengths')
plt.xlabel('Sentence Length (Number of Words)')
plt.ylabel('Frequency')
plt.show()

#90th percentile for setence length
percentile_90 = np.percentile(sentence_lengths, 90)
print(f"90% Percentile Sentence Length: {percentile_90}")
#size of the largest sentence
largest_sentence_length = max(sentence_lengths)
print(f"Largest Sentence Length: {largest_sentence_length}")


# In[6]:


max_sequence_length = largest_sentence_length
print(f"Max Sequence Length: {max_sequence_length}")

# Pad sequences to the maximum sequence length (90th percentile)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')

print(f"Padded Sequences shape: {padded_sequences.shape}")
single_sequence = padded_sequences[0]
print(f"Single padded sequence:\n{single_sequence}")

vocab_size = len(tokenizer.word_index) + 1
print(f"Vocabulary Size: {vocab_size}")


# In[7]:


#SPLIT DATA
X_train, X_val_test, y_train, y_val_test = train_test_split(padded_sequences, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)
print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")


# In[8]:


#NumPy conversion
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
X_val = np.array(X_val)
y_val = np.array(y_val)


# In[9]:


np.save(r"C:\Users\tyler\OneDrive - SNHU\WGU\Neural Networks\Task 2\X_train.npy", X_train)
np.save(r"C:\Users\tyler\OneDrive - SNHU\WGU\Neural Networks\Task 2\X_test.npy", X_test)
np.save(r"C:\Users\tyler\OneDrive - SNHU\WGU\Neural Networks\Task 2\X_val.npy", X_val)
np.save(r"C:\Users\tyler\OneDrive - SNHU\WGU\Neural Networks\Task 2\y_train.npy", y_train)
np.save(r"C:\Users\tyler\OneDrive - SNHU\WGU\Neural Networks\Task 2\y_test.npy", y_test)
np.save(r"C:\Users\tyler\OneDrive - SNHU\WGU\Neural Networks\Task 2\y_val.npy", y_val)


# In[10]:


#Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

#Checkpoint callback to save the best model based on validation loss
checkpoint = ModelCheckpoint('lstm/best_model.keras', 
                             monitor='val_loss', save_best_only=True, mode='min', verbose=1)

#DEFINE MODEL
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100))
model.add(Bidirectional(LSTM(units=64, return_sequences=False)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#BUILD MODEL
model.build(input_shape=(None, max_sequence_length))
model.summary()
#HISTORY
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,
                    batch_size=32,
                    verbose=1,
                    callbacks=[early_stopping, checkpoint])


# In[11]:


best_model = load_model('lstm/best_model.keras')


# In[12]:


#Evaluation
loss, accuracy = best_model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")


# In[17]:


plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
print(f"Final Test Accuracy of Best Model: {accuracy:.4f}")


# In[14]:


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[15]:


#Cacluating predictions on the test set
y_pred = (best_model.predict(X_test) >= 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[18]:


best_model.save('C:/Users/tyler/OneDrive - SNHU/WGU/Neural Networks/Task 2/best_model.keras')

