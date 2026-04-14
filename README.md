How to run it
Open a terminal in the project root:
sqli_xss_detection

Install required Python packages:
python -m pip install tensorflow pandas scikit-learn numpy

Train the model:
cd dataset
python train_model.py --fast

This will read payload_dataset1.csv, train the network, and save:
attack_cnn_lstm.h5
tokenizer.json
label_encoder.pkl

Run prediction:
python predict.py
