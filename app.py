from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import pandas as pd
import numpy as np
import joblib

app = FastAPI(title="AI Parking Intelligence")

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
# LOAD MODELS
# -----------------------------------------------------
DBSCAN_MODEL = joblib.load("model/dbscan.pkl")
SCALER = joblib.load("model/scaler.pkl")

# -----------------------------------------------------
# ROOT (FOR RENDER)
# -----------------------------------------------------
@app.get("/")
def root():
    return {"status": "Backend running"}

# -----------------------------------------------------
# 1️⃣ HOTSPOTS (PLACE BASED – NO MADURAI)
# -----------------------------------------------------
@app.get("/predict/hotspots")
def get_hotspots(
    date: str = None,
    center_lat: float = None,
    center_lng: float = None
):
    # --- fallback center if frontend still uses date ---
    if center_lat is None or center_lng is None:
        center_lat = 12.9716   # neutral default (not Madurai)
        center_lng = 77.5946

    hotspots = []

    for i in range(8):
        raw_score = np.random.uniform(30, 120)
        score = int(SCALER.transform([[raw_score]])[0][0])
        band = "high" if score >= 70 else ("medium" if score >= 40 else "low")

        hotspots.append({
            "id": f"H-{i+1}",
            "latitude": center_lat + np.random.uniform(-0.02, 0.02),
            "longitude": center_lng + np.random.uniform(-0.02, 0.02),
            "score": score,
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
# 2️⃣ HEATMAP (PLACE BASED)
# -----------------------------------------------------
 
@app.get("/heatmap/violations")
def violation_heatmap(
    start_date: str = None,
    end_date: str = None,
    hour: int = None,
    center_lat: float = None,
    center_lng: float = None
):
    # fallback center
    if center_lat is None or center_lng is None:
        center_lat = 12.9716
        center_lng = 77.5946

    points = []

    for _ in range(300):
        points.append({
            "latitude": center_lat + np.random.uniform(-0.03, 0.03),
            "longitude": center_lng + np.random.uniform(-0.03, 0.03)
        })

    return points

# -----------------------------------------------------
# 3️⃣ STATS (UNCHANGED)
# -----------------------------------------------------
@app.get("/stats/hourly")
def hourly_stats():
    return {str(i): np.random.randint(5, 40) for i in range(24)}

@app.get("/stats/offence")
def get_offence_stats(start_date: str, end_date: str):
    return {"status": "success", "data": []}

@app.get("/stats/junctions")
def get_junction_stats(start_date: str, end_date: str):
    return {"status": "success", "data": []}