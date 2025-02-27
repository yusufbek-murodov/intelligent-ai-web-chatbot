import os
import random
import pickle
import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

lemmatizer = WordNetLemmatizer()

words = pickle.load(open('model/words.pkl', 'rb'))
classes = pickle.load(open('model/classes.pkl', 'rb'))
model = load_model('model/chatbot_model.keras')


def clean_up_sentence(sentence):
    ignore_symbols = ['?', '!', '.', ',']
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words if word not in ignore_symbols]

    print(f"Tokenized Sentence: {sentence_words}")
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)

    for w in sentence_words:
        if w in words:
            bag[words.index(w)] = 1

    print(f"Bag of Words: {bag}")
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]

    ERROR_THRESHOLD = 0.15

    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    if results:
        intent = classes[results[0][0]]
        probability = results[0][1]
        print(f"Predicted Intent: {intent} (Probability: {probability:.4f})")
        return [{'intent': intent, 'probability': str(probability)}]
    else:
        return [{'intent': 'unknown', 'probability': '0'}]

def get_response(intents_list):
    try:
        with open('model/intents.json', encoding='utf-8') as file:
            intents_json = json.load(file)

        tag = intents_list[0]['intent']
        for intent in intents_json['intents']:
            if intent['tag'] == tag:
                response = random.choice(intent['responses'])
                print(f"Selected Response: {response}")
                return response

    except Exception as e:
        print(f"Error loading intents.json: {e}")
        return "Sorry, I didn't understand that."

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break

        intents = predict_class(user_input)
        response = get_response(intents)
        print(f"Chatbot: {response}")

