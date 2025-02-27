import os
import pickle
import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


nltk.download('punk')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignore_symbols = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_symbols]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('../Flask Application/model/words.pkl', 'wb'))
pickle.dump(classes, open('../Flask Application/model/classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in document[0] if word not in ignore_symbols]

    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

training = np.array(training, dtype=object)
train_x = np.array([i[0] for i in training], dtype=np.float32)
train_y = np.array([i[1] for i in training], dtype=np.float32)

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('../Flask Application/model/chatbot_model.keras')

print("Model training complete and saved successfully!")
