

# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.svm import LinearSVC
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import joblib

# df = pd.read_csv("payload_dataset.csv")

# # Clean
# df = df.drop_duplicates()
# df = df.dropna()
# df = df[df["payload"].str.strip() != ""]

# # Vectorizer
# tfidf = TfidfVectorizer(
#     analyzer='char',
#     ngram_range=(2, 6),
#     min_df=1
# )
# X = tfidf.fit_transform(df["payload"])
# y = df["label"]

# # Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# # Train SVM
# model = LinearSVC()
# model.fit(X_train, y_train)

# # Evaluate
# y_pred = model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))

# # Save
# joblib.dump(model, "attack_classifier.pkl")
# joblib.dump(tfidf, "vectorizer.pkl")

# ...existing code...
# ...existing code...
import os
import re
import joblib
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy import sparse

# Small engineered numeric features from text
class TextStats(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        rows = []
        for s in X:
            s = "" if s is None else str(s)
            length = len(s)
            digits = sum(c.isdigit() for c in s)
            specials = sum(1 for c in s if not c.isalnum() and not c.isspace())
            has_script = 1 if re.search(r"script|onerror|onload|<iframe|<img|javascript", s, re.I) else 0
            has_time = 1 if re.search(r"\b(sleep|pg_sleep|waitfor|benchmark|randomblob)\b", s, re.I) else 0
            rows.append([length, digits, specials, has_script, has_time])
        return sparse.csr_matrix(np.array(rows, dtype=np.float32))

def load_and_clean(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    df = df.dropna(subset=["payload", "label"])
    df["payload"] = df["payload"].astype(str)
    df = df[df["payload"].str.strip().astype(bool)]
    df["label"] = df["label"].astype(str)
    return df

def print_diagnostics(y):
    cnt = Counter(y)
    print("Total samples:", sum(cnt.values()))
    print("Label counts:", cnt)
    dummy = DummyClassifier(strategy="most_frequent")
    try:
        cv = cross_val_score(dummy, X_raw, y, cv=5, scoring="accuracy")
        print("Dummy (most_frequent) 5-fold accuracy:", float(np.mean(cv)))
    except Exception:
        pass

if __name__ == "__main__":
    DATA_PATH = os.path.join(os.path.dirname(__file__), "payload_dataset.csv")
    print("Loading", DATA_PATH)
    df = load_and_clean(DATA_PATH)

    X_raw = df["payload"].values
    y = df["label"].values

    print_diagnostics(y)

    # Stratified split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.25, random_state=42, stratify=y
    )

    # Build pipeline: char-TF-IDF + numeric stats
    tfidf = TfidfVectorizer(analyzer="char", dtype=np.float32)
    features = FeatureUnion([
        ("tfidf", tfidf),
        ("stats", TextStats())
    ])

    pipe = Pipeline([
        ("features", features),
        ("clf", LinearSVC(max_iter=10000, dual=False))
    ])

    # Quick baseline training with class_weight balanced to check improvement
    quick_pipe = Pipeline([
        ("features", features),
        ("clf", LinearSVC(class_weight="balanced", C=1.0, max_iter=10000, dual=False))
    ])

    print("Fitting quick baseline model (class_weight='balanced')...")
    quick_pipe.fit(X_train_raw, y_train)
    yq = quick_pipe.predict(X_test_raw)
    print("Quick baseline accuracy:", accuracy_score(y_test, yq))
    print(classification_report(y_test, yq))
    print("Confusion matrix:\n", confusion_matrix(y_test, yq))

    # Hyperparameter search (randomized)
    param_dist = {
        "features__tfidf__ngram_range": [(2,4), (2,6), (3,6)],
        "features__tfidf__max_df": [0.75, 0.85, 0.95, 1.0],
        "features__tfidf__min_df": [1, 2, 3],
        "features__tfidf__sublinear_tf": [True, False],
        "features__tfidf__max_features": [20000, 50000, None],
        "clf__C": [0.01, 0.1, 1, 5, 10],
        "clf__class_weight": [None, "balanced"]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=30,
        scoring="f1_macro",
        n_jobs=-1,
        cv=cv,
        verbose=1,
        random_state=42,
        return_train_score=False
    )

    print("Starting hyperparameter search (this may take several minutes)...")
    search.fit(X_train_raw, y_train)

    best = search.best_estimator_
    print("Best params:", search.best_params_)

    # Evaluate best model
    y_pred = best.predict(X_test_raw)
    print("Best model accuracy:", accuracy_score(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # Save artifacts
    MODEL_OUT = os.path.join(os.path.dirname(__file__), "attack_classifier.pkl")
    VEC_OUT = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")
    joblib.dump(best, MODEL_OUT)
    # extract tfidf vectorizer (first transformer)
    try:
        tf = best.named_steps["features"].transformer_list[0][1]
        joblib.dump(tf, VEC_OUT)
    except Exception:
        joblib.dump(tfidf, VEC_OUT)
    print("Saved model to", MODEL_OUT)
    print("Saved vectorizer to", VEC_OUT)
# ...existing code...