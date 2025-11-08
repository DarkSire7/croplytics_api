import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

class SoilData(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float
    location: str


class SoilHealthData(BaseModel):
    n_val: float
    p_val: float
    k_val: float
    ph_val: float


# Initialize FastAPI App
app = FastAPI()

try:
    # M1
    model_m1_classifier = joblib.load("crop_recommender_model.joblib")
    m1_class_mapping = joblib.load("crop_class_mapping.joblib")

    # M3
    model_m3_yield = joblib.load("yield_model.joblib")
    crop_name_to_code = joblib.load("yield_class_mapping.joblib")

    print("All models loaded successfully!")
except FileNotFoundError as e:
    print(f"ERROR: A model file is missing! {e}")
    model_m1_classifier, m1_class_mapping = None, None
    model_m3_yield, crop_name_to_code = None, None

# M1 Filter
REGIONAL_CROP_DATABASE = {
    "Andhra Pradesh": [
        "banana",
        "blackgram",
        "chickpea",
        "coconut",
        "cotton",
        "grapes",
        "jute",
        "maize",
        "mango",
        "mungbean",
        "muskmelon",
        "papaya",
        "pomegranate",
        "rice",
        "watermelon",
    ],
    "Arunachal Pradesh": ["apple", "orange", "rice"],
    "Assam": ["jute", "orange", "rice", "banana", "papaya"],
    "Bihar": [
        "jute",
        "lentil",
        "maize",
        "mango",
        "muskmelon",
        "rice",
        "watermelon",
        "pigeonpeas",
        "chickpea",
    ],
    "Chhattisgarh": ["muskmelon", "rice", "maize", "chickpea", "lentil"],
    "Gujarat": [
        "banana",
        "chickpea",
        "coconut",
        "cotton",
        "mango",
        "mothbeans",
        "papaya",
        "pigeonpeas",
        "pomegranate",
        "mungbean",
    ],
    "Haryana": ["cotton", "mothbeans", "muskmelon", "rice", "watermelon", "maize"],
    "Himachal Pradesh": ["apple", "kidneybeans", "rice", "maize", "mango"],
    "Jammu & Kashmir": ["apple", "kidneybeans", "mothbeans", "rice", "maize", "mango"],
    "Jharkhand": ["lentil", "rice", "maize", "pigeonpeas", "blackgram"],
    "Karnataka": [
        "banana",
        "coconut",
        "coffee",
        "cotton",
        "grapes",
        "kidneybeans",
        "maize",
        "mango",
        "mungbean",
        "papaya",
        "pigeonpeas",
        "pomegranate",
        "rice",
        "watermelon",
        "chickpea",
    ],
    "Kerala": ["coconut", "coffee", "kidneybeans", "rice", "banana", "papaya"],
    "Madhya Pradesh": [
        "blackgram",
        "chickpea",
        "lentil",
        "maize",
        "mothbeans",
        "muskmelon",
        "orange",
        "pigeonpeas",
        "pomegranate",
        "cotton",
    ],
    "Maharashtra": [
        "banana",
        "blackgram",
        "chickpea",
        "cotton",
        "grapes",
        "kidneybeans",
        "maize",
        "mothbeans",
        "mungbean",
        "muskmelon",
        "orange",
        "papaya",
        "pigeonpeas",
        "pomegranate",
        "watermelon",
        "rice",
    ],
    "Nagaland": ["apple", "rice", "maize"],
    "Odisha": ["coconut", "jute", "mungbean", "rice", "watermelon", "maize"],
    "Punjab": [
        "cotton",
        "grapes",
        "kidneybeans",
        "mothbeans",
        "muskmelon",
        "orange",
        "rice",
        "watermelon",
        "maize",
        "mango",
    ],
    "Rajasthan": [
        "blackgram",
        "chickpea",
        "cotton",
        "lentil",
        "mothbeans",
        "mungbean",
        "muskmelon",
        "orange",
        "pomegranate",
        "maize",
    ],
    "Tamil Nadu": [
        "banana",
        "blackgram",
        "coconut",
        "coffee",
        "grapes",
        "kidneybeans",
        "maize",
        "mungbean",
        "muskmelon",
        "rice",
        "watermelon",
        "papaya",
        "mango",
        "pomegranate",
    ],
    "Telangana": [
        "cotton",
        "maize",
        "mango",
        "pigeonpeas",
        "rice",
        "watermelon",
        "orange",
        "mungbean",
    ],
    "Uttar Pradesh": [
        "blackgram",
        "chickpea",
        "lentil",
        "maize",
        "mango",
        "mothbeans",
        "muskmelon",
        "pigeonpeas",
        "rice",
        "watermelon",
        "banana",
        "papaya",
    ],
    "Uttarakhand": ["apple", "kidneybeans", "rice", "maize", "mango", "lentil"],
    "West Bengal": [
        "coconut",
        "jute",
        "kidneybeans",
        "lentil",
        "mango",
        "rice",
        "watermelon",
        "maize",
        "papaya",
    ],
}

# API Endpoints


def get_regional_shortlist(state: str):
    """Step 1.1: Get list of regionally-viable crops."""
    global m1_class_mapping
    all_crops = list(m1_class_mapping.values())
    return REGIONAL_CROP_DATABASE.get(state, all_crops)


@app.post("/recommend-crop")
def get_recommendation(data: SoilData):
    """
    This is the main endpoint that runs the M1 -> M3 pipeline.
    """

    # REGIONAL FILTER
    regional_crops = get_regional_shortlist(data.location)

    # Top 3 "Likely" Crops

    # non-scaled data for M1
    input_df = pd.DataFrame([data.dict()])
    input_df_for_m1 = input_df.drop("location", axis=1)

    all_probabilities = model_m1_classifier.predict_proba(input_df_for_m1)[0]

    # Map scores to crop names
    all_crop_scores = {}
    for i, score in enumerate(all_probabilities):
        crop_name = m1_class_mapping[i]
        all_crop_scores[crop_name] = score

    # Filter by region
    regional_scores = {}
    for crop in regional_crops:
        if crop in all_crop_scores:
            regional_scores[crop] = all_crop_scores[crop]

    # Sort by M1's score
    sorted_by_m1_score = sorted(
        regional_scores.items(), key=lambda item: item[1], reverse=True
    )

    # final shortlist of 3
    shortlist = [crop for crop, score in sorted_by_m1_score[:3]]
    # Get Yield Rank
    final_rankings = []

    # Non-scaled data for the M3 model
    input_df_for_m3 = input_df.drop("location", axis=1)

    for crop in shortlist:
        m3_input = input_df_for_m3.copy()

        m3_input["crop_code"] = list(crop_name_to_code.keys())[
            list(crop_name_to_code.values()).index(crop)
        ]

        # Predict the yield
        predicted_yield = model_m3_yield.predict(m3_input)[0]

        # Add to final list
        final_rankings.append(
            {
                "crop": crop,
                "expected_yield_quintals_per_acre": round(float(predicted_yield), 2),
            }
        )

    # Sort the final list by the *highest yield*
    sorted_by_yield = sorted(
        final_rankings,
        key=lambda k: k["expected_yield_quintals_per_acre"],
        reverse=True,
    )

    return {"ranked_recommendations": sorted_by_yield}


@app.post("/analyze-soil-health")
def get_soil_health(data: SoilHealthData):
    """
    This is the M2 Soil Health Analyzer (Rule-Based).
    """
    issues = []

    # pH Analysis
    if data.ph_val < 6.0:
        issues.append(f"Soil is too acidic (pH: {data.ph_val}).")
    elif data.ph_val > 7.5:
        issues.append(f"Soil is too alkaline (pH: {data.ph_val}).")

    # Nitrogen (N) Analysis (using official Indian guidelines)
    if data.n_val < 280:
        issues.append(f"Nitrogen (N) is low ({data.n_val} kg/ha).")

    # Phosphorus (P) Analysis
    if data.p_val < 10:
        issues.append(f"Phosphorus (P) is very low ({data.p_val} kg/ha).")

    # Potassium (K) Analysis
    if data.k_val < 120:
        issues.append(f"Potassium (K) is low ({data.k_val} kg/ha).")

    # Final Classification
    if len(issues) == 0:
        return {
            "status": "Good",
            "color_code": "green",
            "reason": "All major nutrients and pH are in a good range.",
        }
    else:
        return {
            "status": "Needs Improvement",
            "color_code": "yellow",
            "reason": " ".join(issues),
        }


@app.get("/")
def read_root():
    return {"message": "Croplytics ML API is running!"}
