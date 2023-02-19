import os
import tensorflow as tf
from flask import Flask, jsonify, request, make_response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

import matplotlib.pyplot as plt

class TrainingVisualizer(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accs = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accs.append(logs.get('accuracy'))
        plt.plot(range(len(self.losses)), self.losses, 'b')
        plt.plot(range(len(self.accs)), self.accs, 'r')
        plt.title('Training Loss and Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Loss', 'Accuracy'], loc='upper left')
        plt.show()

# Chargement des données d'entraînement
training_data = open("filtered_training_data.txt", "r", encoding='utf8').read()
corpus = training_data.lower().split("\n")

# Création d'un tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# Préparation des données d'entraînement
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Remplissage des séquences pour qu'elles aient toutes la même longueur
max_sequence_len = max([len(seq) for seq in input_sequences])
input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

# Séparation des données en entrée (X) et en sortie (y)
X = input_sequences[:,:-1]
y = input_sequences[:,-1]

# Conversion des étiquettes de sortie en format catégoriel
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Définition du modèle de langage naturel
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(tf.keras.layers.LSTM(100, return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(100))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(total_words, activation='softmax'))

def train_model():
    # Compilation et entraînement du modèle
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X, y, epochs=150, verbose=1)

    # Affichage des courbes de perte et d'accuracy
    plt.plot(history.history['loss'])
    plt.plot(history.history['accuracy'])
    plt.title('Evolution de la perte et de l\'accuracy pendant l\'entraînement')
    plt.xlabel('Epoch')
    plt.legend(['Perte', 'Accuracy'], loc='upper right')
    plt.show()

    # Sauvegarde du modèle dans un fichier
    if not os.path.exists("saved_model"):
        os.mkdir("saved_model")
    model.save("saved_model/nlp_model")

def predict():
    # Chargement du modèle depuis le fichier
    model = tf.keras.models.load_model("saved_model/nlp_model")

    seed_text = str(input("prompt => "))
    next_words = int(input("lenght => "))

    # Chargement des données d'entraînement
    with open("training_data.txt", "a", encoding='utf8') as f:
        f.write(f'\n{seed_text}')
        f.close()

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_class = tf.argmax(predicted, axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_class:
                output_word = word
                break
        seed_text += " " + output_word

    response = {'generated_text': seed_text}

    print(response)

while True:
    print("Choisissez une option :")
    print("1. Entraîner le modèle")
    print("2. Utiliser le modèle")
    print("3. Quitter")

    choice = input("Entrez votre choix : ")

    if choice == "1":
        train_model()
    elif choice == "2":
       predict()
    elif choice == "3":
        break
    else:
        print("Choix invalide. Veuillez entrer un choix valide.")