from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib

app = FastAPI(title="AI Parking Intelligence")

# -----------------------------------------------------
# LOAD MODELS (SAFE)
# -----------------------------------------------------
DBSCAN_MODEL = joblib.load("model/dbscan.pkl")
SCALER = joblib.load("model/scaler.pkl")

# -----------------------------------------------------
# CORS
# -----------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------
# 🔥 MOCK DATAFRAME (REPLACES CSV SAFELY)
# -----------------------------------------------------
np.random.seed(42)

rows = 500
base_date = datetime(2023, 11, 20)

df = pd.DataFrame({
    "created_datetime": [
        base_date + timedelta(hours=np.random.randint(0, 24))
        for _ in range(rows)
    ],
    "latitude": 9.92 + np.random.uniform(-0.03, 0.03, rows),
    "longitude": 78.11 + np.random.uniform(-0.03, 0.03, rows),
    "junction_name": np.random.choice(
        ["KK Nagar", "Anna Nagar", "Periyar", "Goripalayam", "Mattuthavani"],
        rows
    ),
    "vehicle_type": np.random.choice(
        ["CAR", "SCOOTER", "MAXI-CAB", "LCV"],
        rows
    ),
    "violation_type": np.random.choice(
        ["NO PARKING", "MAIN ROAD", "ROAD CROSSING"],
        rows
    )
})

df["hour"] = df["created_datetime"].dt.hour

# -----------------------------------------------------
# 1️⃣ HOTSPOTS (UNCHANGED RESPONSE)
# -----------------------------------------------------
@app.get("/predict/hotspots")
def get_hotspots():
    hotspots = []

    for i in range(8):
        raw_score = np.random.uniform(30, 120)
        score = float(SCALER.transform([[raw_score]])[0][0])

        band = "high" if score >= 70 else ("medium" if score >= 40 else "low")

        hotspots.append({
            "id": f"H-{i+1}",
            "latitude": 9.92 + np.random.uniform(-0.02, 0.02),
            "longitude": 78.11 + np.random.uniform(-0.02, 0.02),
            "score": int(score),
            "band": band,
            "priority": "P1" if band == "high" else "P2",
            "peak": "09:00–12:00",
            "violationFrequency": np.random.randint(10, 120),
            "dominantVehicle": "CAR",
            "dominantViolation": "NO PARKING"
        })

    return {
        "total_hotspots": len(hotspots),
        "hotspots": hotspots
    }

# -----------------------------------------------------
# 2️⃣ HEATMAP (UNCHANGED RESPONSE)
# -----------------------------------------------------
@app.get("/heatmap/violations")
def violation_heatmap():
    return df[["latitude", "longitude"]].to_dict(orient="records")

# -----------------------------------------------------
# 3️⃣ STATS (UNCHANGED RESPONSE)
# -----------------------------------------------------
@app.get("/stats/hourly")
def hourly_stats(date: str = Query("2023-11-20")):
    try:
        d = datetime.strptime(date, "%Y-%m-%d").date()
    except Exception:
        return {}

    temp = df[df["created_datetime"].dt.date == d]
    hourly = temp["hour"].value_counts().sort_index()
    return hourly.to_dict()

@app.get("/stats/offence")
async def get_offence_stats(start_date: str = Query(...), end_date: str = Query(...)):
    return {"status": "success", "data": []}

@app.get("/stats/junctions")
async def get_junction_stats(start_date: str = Query(...), end_date: str = Query(...)):
    return {"status": "success", "data": []}