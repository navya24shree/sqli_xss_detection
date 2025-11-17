import os
import random
import json
import pickle
import numpy as np
import pandas as pd
from collections import Counter

# set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample, class_weight
from sklearn.metrics import classification_report, confusion_matrix

# Configurable params
DATA_PATH = os.path.join(os.path.dirname(__file__), "payload_dataset.csv")
MODEL_OUT = os.path.join(os.path.dirname(__file__), "attack_cnn_lstm.h5")
TOKENIZER_OUT = os.path.join(os.path.dirname(__file__), "tokenizer.json")
LABEL_ENCODER_OUT = os.path.join(os.path.dirname(__file__), "label_encoder.pkl")

MAX_NUM_WORDS = 20000    # vocabulary size for tokenizer (token-level)
MAX_SEQUENCE_LENGTH = 150
EMBEDDING_DIM = 128
BATCH_SIZE = 32
EPOCHS = 50

def load_and_clean(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    df = df.dropna(subset=["payload", "label"])
    df["payload"] = df["payload"].astype(str).str.strip()
    df = df[df["payload"].astype(bool)]
    df["label"] = df["label"].astype(str)
    return df

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

def build_model(vocab_size, maxlen, embedding_dim, num_classes):
    inp = layers.Input(shape=(maxlen,), dtype="int32")
    x = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen)(inp)
    x = layers.Conv1D(filters=256, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=False, dropout=0.4, recurrent_dropout=0.2))(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(learning_rate=5e-4)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    print("Loading data from", DATA_PATH)
    df = load_and_clean(DATA_PATH)
    X = df["payload"].values
    y = df["label"].values
    print("Total samples:", len(y), "Label counts:", dict(Counter(y)))

    # Label encode
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    with open(LABEL_ENCODER_OUT, "wb") as f:
        pickle.dump(le, f)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.25, random_state=SEED, stratify=y_enc)

    # Upsample training set to balance classes
    X_train, y_train = upsample_train(X_train, y_train)
    print("After upsampling train label counts:", dict(Counter(y_train)))

    # Tokenizer (token-level)
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)  # fit only on training text
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")

    # Save tokenizer
    token_json = tokenizer.to_json()
    with open(TOKENIZER_OUT, "w", encoding="utf-8") as f:
        f.write(token_json)

    # Compute class weights on upsampled set (if still imbalance)
    cw = class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    class_weights = {i: cw_val for i, cw_val in enumerate(cw)}
    print("Class weights:", class_weights)

    num_classes = len(le.classes_)
    vocab_size = min(MAX_NUM_WORDS, len(tokenizer.word_index) + 1)

    model = build_model(vocab_size=vocab_size, maxlen=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM, num_classes=num_classes)
    model.summary()

    # Callbacks
    es = callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1)
    rl = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
    ck = callbacks.ModelCheckpoint(MODEL_OUT, monitor="val_loss", save_best_only=True, verbose=1)

    # Train
    history = model.fit(
        X_train_pad,
        y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[es, rl, ck],
        verbose=2
    )

    # Evaluate
    print("Evaluating on test set...")
    loss, acc = model.evaluate(X_test_pad, y_test, verbose=0)
    print(f"Test loss: {loss:.4f}  Test accuracy: {acc:.4f}")

    # Predictions and detailed report
    y_pred_probs = model.predict(X_test_pad)
    y_pred = np.argmax(y_pred_probs, axis=1)
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save final model (best saved by checkpoint) and objects
    # checkpoint already saved best model; ensure one final save
    model.save(MODEL_OUT)
    print("Saved model to", MODEL_OUT)
    print("Saved tokenizer to", TOKENIZER_OUT)
    print("Saved label encoder to", LABEL_ENCODER_OUT)

    # Note: reaching 94% accuracy on this small dataset may be unrealistic without more data/augmentation.
# filepath: c:\Users\PC\OneDrive\Documents\mini