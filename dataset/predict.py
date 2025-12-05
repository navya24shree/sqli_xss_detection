import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# --- EXACT COPY of AttentionLayer from train_model.py ---
class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        score = tf.nn.softmax(tf.reduce_sum(x, axis=2), axis=1)
        score = tf.expand_dims(score, axis=2)
        context = x * score
        return tf.reduce_sum(context, axis=1)

    def get_config(self):
        cfg = super().get_config()
        return cfg

# --- paths & constants ---
MODEL_PATH = "attack_cnn_lstm.h5"
TOKENIZER_PATH = "tokenizer.json"
LABEL_ENCODER_PATH = "label_encoder.pkl"
MAX_SEQUENCE_LENGTH = 120  # match training

# load model with custom_objects
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"AttentionLayer": AttentionLayer})

# load tokenizer
with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
    tokenizer_json = f.read()
tokenizer = tokenizer_from_json(tokenizer_json)

# load label encoder
with open(LABEL_ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

def predict_payload(text: str) -> str:
    """Predict attack type for a payload"""
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
    probs = model.predict(pad, verbose=0)
    pred = int(np.argmax(probs, axis=1)[0])
    return le.classes_[pred]

if __name__ == "__main__":
    print("Test 1", predict_payload("' OR '1"))
