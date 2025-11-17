import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Load model, tokenizer, encoder
# -----------------------------
MODEL_PATH = "attack_cnn_lstm.h5"  # MUST match your saved model filename
TOKENIZER_PATH = "tokenizer.json"
LABEL_ENCODER_PATH = "label_encoder.pkl"

# This must match the training script
MAX_SEQUENCE_LENGTH = 120  

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load tokenizer
with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
    tokenizer_data = json.load(f)

tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json.dumps(tokenizer_data))

# Load label encoder
with open(LABEL_ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

# -----------------------------
# Prediction function
# -----------------------------
def predict_payload(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")

    probs = model.predict(pad)
    pred = np.argmax(probs, axis=1)[0]

    return le.classes_[pred]


# -----------------------------
# Test Prediction
# -----------------------------
# print(predict_payload("<image/src/onerror=prompt(8)>"))
print("xss\n",predict_payload("1 OR '1'='1'; --"))
print(predict_payload("hello world"))
