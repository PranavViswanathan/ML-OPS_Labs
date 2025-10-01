from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# ---------- Load Model ----------
MODEL_PATH = "model/titanic_gb.pkl"
model = joblib.load(MODEL_PATH)

# ---------- Define API ----------
app = FastAPI(title="Titanic Survival Prediction API (Scikit-Learn)")

class Passenger(BaseModel):
    pclass: int
    sex: int      # 0 = male, 1 = female
    age: float
    sibsp: int
    parch: int
    fare: float
    embarked: int # 0 = C, 1 = Q, 2 = S

@app.get("/")
def home():
    return {"message": "Titanic Survival Prediction API is running!"}

@app.post("/predict")
def predict(passenger: Passenger):
    try:
        data = np.array([
            passenger.pclass,
            passenger.sex,
            passenger.age,
            passenger.sibsp,
            passenger.parch,
            passenger.fare,
            passenger.embarked
        ]).reshape(1, -1)

        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0][1]

        return {
            "survived": int(prediction),
            "probability_of_survival": float(probability)
        }
    except Exception as e:
        return {"error": str(e)}
