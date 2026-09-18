import os
import pickle

from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# --------------------------------------------------
# MODEL
# --------------------------------------------------

MODEL_PATH = os.path.join("models", "pokemon_model.pkl")


# --------------------------------------------------
# LOAD SAVED FINAL MODEL
# --------------------------------------------------

def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "pokemon_model.pkl not found in the models folder."
        )

    with open(MODEL_PATH, "rb") as file:
        saved_data = pickle.load(file)

    model = saved_data["model"]
    feature_columns = saved_data["features"]

    return model, feature_columns


model, feature_columns = load_model()


# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        try:

            user_input = {
                "hp": float(request.form["hp"]),
                "attack": float(request.form["attack"]),
                "defense": float(request.form["defense"]),
                "height_dm": float(request.form["height_dm"]),
                "weight_hg": float(request.form["weight_hg"]),
                "base_experience": float(
                    request.form["base_experience"]
                ),
                "type_2": request.form["type_2"]
            }

            # Create input DataFrame
            input_df = pd.DataFrame([user_input])

            # One-hot encode type_2
            input_df = pd.get_dummies(
                input_df,
                columns=["type_2"],
                drop_first=False
            )

            # Match exactly the columns used during training
            input_df = input_df.reindex(
                columns=feature_columns,
                fill_value=0
            )

            # Predict
            prediction = model.predict(input_df)[0]

            return render_template(
                "result.html",
                prediction=prediction,
                data=user_input
            )

        except ValueError:
            return "Please enter valid numeric values."

        except Exception as e:
            return f"Prediction error: {e}"

    return render_template("predict.html")


# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
    