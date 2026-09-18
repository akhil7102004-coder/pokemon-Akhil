import os
import pickle

from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

MODEL_PATH = os.path.join("models", "pokemon_pipeline.pkl")
FEATURE_COLUMNS = [
    "hp",
    "attack",
    "defense",
    "height_dm",
    "weight_hg",
    "base_experience",
]


def train_model():
    df = pd.read_csv("G2.csv")

    needed_columns = FEATURE_COLUMNS + ["type_1"]

    df = df[needed_columns].dropna()
    df["base_experience"] = df["base_experience"].fillna(df["base_experience"].median())

    X = df[FEATURE_COLUMNS]
    y = df["type_1"]

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    max_depth=20,
                    min_samples_leaf=1,
                ),
            ),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


def save_model(model):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(
            {
                "pipeline": model,
                "features": FEATURE_COLUMNS,
                "target": "type_1",
            },
            file,
        )


def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as file:
            saved_data = pickle.load(file)
        return saved_data["pipeline"], saved_data["features"]

    model = train_model()
    save_model(model)
    return model, FEATURE_COLUMNS


model, feature_columns = load_model()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        user_input = {
            "hp": float(request.form["hp"]),
            "attack": float(request.form["attack"]),
            "defense": float(request.form["defense"]),
            "height_dm": float(request.form["height_dm"]),
            "weight_hg": float(request.form["weight_hg"]),
            "base_experience": float(request.form["base_experience"]),
        }

        features = pd.DataFrame([user_input], columns=feature_columns)
        prediction = model.predict(features)[0]

        return render_template("result.html", prediction=prediction, data=user_input)

    return render_template("predict.html")


if __name__ == "__main__":
    app.run(debug=True)
