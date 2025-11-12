# Croplytics ML API

This repository contains the backend machine learning API for the Croplytics application.

This service is a FastAPI application that runs a two-stage ML pipeline to provide intelligent crop recommendations based on soil conditions, climate, and geographic location.

## Core Features

* **Rule-Based Regional Filter:** Pre-filters a list of viable crops based on the user's state in India.
* **M1: XGBoost Shortlister:** A classifier model that identifies the top 3 most scientifically compatible crops from the regional list.
* **M3: XGBoost Yield Ranker:** A regression model that predicts the expected yield (in quintals/acre) for each of the top 3 crops.
* **M2: Soil Health Analyzer:** A rule-based engine that classifies soil as "Good" or "Needs Improvement" based on N, P, K, and pH levels.

## Tech Stack

* **API:** Python, FastAPI
* **ML Models:** XGBoost, Pandas, Scikit-learn (sklearn)
* **Deployment:** Docker, Google Cloud Run

---

## How to Run This Project Locally

This API is designed to be run as a service. You can run it on your own machine by following these steps.

### 1. Prerequisites

* Python 3.10+
* Git and [Git LFS](https://git-lfs.com/) (for pulling the large `.joblib` model files)

### 2. Clone the Repository

You must use `git clone` (not download ZIP) to get the model files handled by Git LFS.

```bash
# Clone the repo
git clone [https://github.com/Darksire7/croplytics_api.git](https://github.com/Darksire7/croplytics_api.git)
cd croplytics_api
```

### 3. Set Up the Environment

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Mac/Linux
    .\venv\Scripts\activate   # On Windows
    ```

2.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

### 4. Run the API Server

1.  **Run the Uvicorn server:**
    ```bash
    uvicorn main:app --reload
    ```

2.  **You're live!** The API is now running on your local machine at `http://127.0.0.1:8000`.

---

## API Endpoints

You can access the interactive documentation by going to `http://127.0.0.1:8000/docs` in your browser.

### 1. `/recommend-crop` (POST)

Provides a ranked list of the top 3 crop recommendations based on soil/climate data, ranked by the highest expected yield.

**Example Request Body:**
```json
{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 21.5,
  "humidity": 82,
  "ph": 6.5,
  "rainfall": 202,
  "location": "Maharashtra"
}
```

**Example Response Body:**
```json
{
  "ranked_recommendations": [
    {
      "crop": "rice",
      "expected_yield_quintals_per_acre": 21.45
    },
    {
      "crop": "maize",
      "expected_yield_quintals_per_acre": 17.38
    },
    {
      "crop": "banana",
      "expected_yield_quintals_per_acre": 9.33
    }
  ]
}
```

### 2. `/analyze-soil-health` (POST)

Provides a rule-based analysis of the soil's health.

**Example Request Body:**
```json
{
  "n_val": 100,
  "p_val": 25,
  "k_val": 200,
  "ph_val": 5.5
}
```

**Example Response Body:**
```json
{
  "status": "Needs Improvement",
  "color_code": "yellow",
  "reason": "Soil is too acidic (pH: 5.5). Nitrogen (N) is low (100 kg/ha)."
}
```