# CNN+LSTM Payload Classifier

This project trains a CNN+LSTM neural network to classify payloads as `sqli`, `xss`, or `benign` using token-level sequences.

## Requirements

- Python 3.8+
- pip packages:
  - tensorflow
  - pandas
  - numpy
  - scikit-learn

Install requirements (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install tensorflow pandas numpy scikit-learn
```

## How to Run

1. Place your dataset file as `payload_dataset.csv` in the same folder as `train_model.py`.
2. Open a terminal in this folder.
3. Run the training script:

```powershell
python train_model.py
```

The script will:
- Load and clean the dataset
- Upsample classes for balance
- Tokenize and pad payloads
- Train a CNN+LSTM model
- Save the trained model (`attack_cnn_lstm.h5`), tokenizer (`tokenizer.json`), and label encoder (`label_encoder.pkl`)

## Output

- `attack_cnn_lstm.h5` — trained Keras model
- `tokenizer.json` — saved tokenizer for preprocessing
- `label_encoder.pkl` — label encoder for decoding predictions

## Notes

- Accuracy depends on dataset size and quality. For best results, use more labeled data.
- To use the model for prediction, load the model, tokenizer, and label encoder in your inference script.
