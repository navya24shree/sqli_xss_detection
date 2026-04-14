

import os # used for working with files and folder paths
import random  # generartes random values
import json  # used for reading and writing json files
import pickle   # used to save python objects
import argparse  # parse command-line options
import numpy as np   # used for numerical operations
import pandas as pd   #used for reading csv files and data cleaning
from collections import Counter  #Used to count occurrences of labels.

# set seeds for reproducibility
#Reproducibility means that every time you run your code, you get the same results.
#Creating new synthetic (fake but realistic) training samples from your existing data.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf #It is deep learning framework used to build, train, and save your neural network model.
from tensorflow.keras import layers, models, callbacks, optimizers  # in model architecture and training.
from tensorflow.keras.preprocessing.text import Tokenizer  #converts text → integer sequence.
from tensorflow.keras.preprocessing.sequence import pad_sequences  #make all sequences same length

from sklearn.model_selection import train_test_split #Splits your dataset into Training set and Testing set
from sklearn.preprocessing import LabelEncoder #converts labels(sqli,xss,benign) to integers
from sklearn.utils import resample, class_weight #To balance the dataset.
from sklearn.metrics import classification_report, confusion_matrix  #to evaluate model performance

# Configurable paths
DATA_PATH = os.path.join(os.path.dirname(__file__), "payload_dataset1.csv")
MODEL_OUT = os.path.join(os.path.dirname(__file__), "attack_cnn_lstm.h5")
TOKENIZER_OUT = os.path.join(os.path.dirname(__file__), "tokenizer.json")
LABEL_ENCODER_OUT = os.path.join(os.path.dirname(__file__), "label_encoder.pkl")

# Default hyperparameters
DEFAULT_MAX_NUM_WORDS = 20000
DEFAULT_MAX_SEQUENCE_LENGTH = 120
DEFAULT_EMBEDDING_DIM = 128
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 40

#load & clean data- remove dupliactes and remove empty payloads
def load_and_clean(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    df = df.dropna(subset=["payload", "label"])
    df["payload"] = df["payload"].astype(str).str.strip()
    df = df[df["payload"].astype(bool)]
    df["label"] = df["label"].astype(str)
    return df

#to balamce the dataset by upsampling minority classes
def upsample_train(X_train, y_train):
    train_df = pd.DataFrame({"text": X_train, "label": y_train})
    max_n = train_df["label"].value_counts().max()
    parts = []

    for lbl, grp in train_df.groupby("label"):
        if len(grp) < max_n:
            parts.append(resample(grp, replace=True, n_samples=max_n, random_state=SEED))
        else:
            parts.append(grp)

    balanced = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return balanced["text"].values, balanced["label"].values


#ATTENTION LAYER (Lightweight)
#It is designed to let the model focus on the most important parts of the sequence (tokens) instead of treating every token equally.
class AttentionLayer(layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        score = tf.nn.softmax(tf.reduce_sum(x, axis=2), axis=1)
        score = tf.expand_dims(score, axis=2)
        context = x * score
        return tf.reduce_sum(context, axis=1)


# OPTIMIZED MODEL

def build_model(vocab_size, maxlen, embedding_dim, num_classes, conv_filters=256, lstm_units=128):
    
    # accepts input sequence after tokenisation
    inp = layers.Input(shape=(maxlen,), dtype="int32")

    # Embedding - Converts each token (number) into a dense vector representation.
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=maxlen,
        mask_zero=True
    )(inp)

    # BETTER REGULARIZATION FOR TEXT - Makes the model more stable when training text data.
    x = layers.SpatialDropout1D(0.3)(x)

    # MULTI-SCALE CNN FOR BETTER PATTERN CAPTURE
    conv3 = layers.Conv1D(conv_filters, kernel_size=3, padding="same", activation="relu")(x) #short patterns
    conv5 = layers.Conv1D(conv_filters, kernel_size=5, padding="same", activation="relu")(x)  #medium patterns
    conv7 = layers.Conv1D(conv_filters, kernel_size=7, padding="same", activation="relu")(x)  #long patterns

    x = layers.Concatenate()([conv3, conv5, conv7])
    x = layers.MaxPooling1D(pool_size=2)(x)

    # STRONGER BiLSTM
    x = layers.Bidirectional(
        layers.LSTM(
            lstm_units,
            return_sequences=True,
            dropout=0.3,
            recurrent_dropout=0.2
        )
    )(x)

    # LIGHTWEIGHT ATTENTION
    att = AttentionLayer()(x)

    # FINAL DENSE LAYERS
    x = layers.Dense(128, activation="relu")(att)
    x = layers.Dropout(0.4)(x)

    out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inp, outputs=out)

    opt = optimizers.Adam(learning_rate=4e-4)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# MAIN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the SQLi/XSS detection model.")
    parser.add_argument("--fast", action="store_true", help="Use a smaller model and fewer epochs for faster training.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size.")
    parser.add_argument("--max-words", type=int, default=DEFAULT_MAX_NUM_WORDS, help="Tokenizer vocabulary size.")
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM, help="Embedding dimension size.")
    args = parser.parse_args()

    max_sequence_length = DEFAULT_MAX_SEQUENCE_LENGTH

    if args.fast:
        print("Fast mode enabled: reducing model size and epochs for quicker execution.")
        epochs = 8
        batch_size = 64
        max_num_words = min(args.max_words, 10000)
        embedding_dim = min(args.embedding_dim, 64)
        conv_filters = 128
        lstm_units = 64
    else:
        epochs = args.epochs
        batch_size = args.batch_size
        max_num_words = args.max_words
        embedding_dim = args.embedding_dim
        conv_filters = 256
        lstm_units = 128

    print("Loading data from", DATA_PATH)
    df = load_and_clean(DATA_PATH)

    X = df["payload"].values
    y = df["label"].values
    print("Total samples:", len(y), "Label counts:", dict(Counter(y)))

    # Label encode - Payload labels(sqli,xss,benign) are converted to integers
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    with open(LABEL_ENCODER_OUT, "wb") as f:
        pickle.dump(le, f)

    # Split + upsample
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.25, random_state=SEED, stratify=y_enc
    )

    X_train, y_train = upsample_train(X_train, y_train)
    print("After upsampling:", dict(Counter(y_train)))

    # Tokenizer- Converts the payload text into numeric sequences
    tokenizer = Tokenizer(num_words=max_num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=max_sequence_length, padding="post")
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_sequence_length, padding="post")

    # Save tokenizer
    with open(TOKENIZER_OUT, "w", encoding="utf-8") as f:
        f.write(tokenizer.to_json())

    # Class weights
    cw = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = {i: cw_val for i, cw_val in enumerate(cw)}

    num_classes = len(le.classes_)
    vocab_size = min(max_num_words, len(tokenizer.word_index) + 1)

    # Build improved model
    model = build_model(
        vocab_size=vocab_size,
        maxlen=max_sequence_length,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        conv_filters=conv_filters,
        lstm_units=lstm_units
    )

    model.summary()

    # Callbacks
    es = callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    rl = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.4, patience=3)
    ck = callbacks.ModelCheckpoint(MODEL_OUT, monitor="val_loss", save_best_only=True)

    # Train
    history = model.fit(
        X_train_pad,
        y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=[es, rl, ck],
        verbose=2
    )

    print("Evaluating...")
    loss, acc = model.evaluate(X_test_pad, y_test, verbose=0)
    print(f"TEST LOSS: {loss:.4f}   TEST ACC: {acc:.4f}")

    # Predictions
    y_pred_probs = model.predict(X_test_pad)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, digits=4))
    
    #confusion matrix tells you which predictions were correct and where the model is failing.
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    model.save(MODEL_OUT)
    print("Model, tokenizer, label encoder saved.")