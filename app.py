from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import json

app = FastAPI(title="AI Parking Intelligence")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------
# LOAD PRECOMPUTED HOTSPOTS AT STARTUP (~KBs not MBs)
# -----------------------------------------------------
with open("model/hotspots.json") as f:
    HOTSPOTS = json.load(f)

# Precompute numpy arrays for fast distance filtering
HS_LATS = np.array([h["latitude"] for h in HOTSPOTS])
HS_LNGS = np.array([h["longitude"] for h in HOTSPOTS])

EARTH_RADIUS_KM = 6371

# -----------------------------------------------------
# HELPERS
# -----------------------------------------------------
def haversine_km(lat1, lng1, lats2, lngs2):
    lat1, lng1 = np.radians(lat1), np.radians(lng1)
    lats2, lngs2 = np.radians(lats2), np.radians(lngs2)
    dlat = lats2 - lat1
    dlng = lngs2 - lng1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lats2) * np.sin(dlng/2)**2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))

def score_to_band(score):
    if score >= 70:
        return "high"
    elif score >= 40:
        return "medium"
    return "low"

# -----------------------------------------------------
# ROOT
# -----------------------------------------------------
@app.get("/")
def root():
    return {"status": "Backend running", "hotspots_loaded": len(HOTSPOTS)}

# -----------------------------------------------------
# 1️⃣ HOTSPOTS — from precomputed JSON, no CSV
# -----------------------------------------------------
@app.get("/predict/hotspots")
def get_hotspots(
    center_lat: float = 9.9252,
    center_lng: float = 78.1198,
    radius_km: float = 5.0
):
    dists = haversine_km(center_lat, center_lng, HS_LATS, HS_LNGS)
    nearby_idx = np.where(dists <= radius_km)[0]

    results = []
    for i in nearby_idx:
        h = HOTSPOTS[i]
        score = h["score"]
        band = score_to_band(score)
        peak = h["peak_hour"]

        results.append({
            "id": f"H-{h['cluster']}",
            "latitude": h["latitude"],
            "longitude": h["longitude"],
            "score": score,
            "band": band,
            "priority": "P1" if band == "high" else "P2",
            "peak": f"{peak:02d}:00–{(peak+3)%24:02d}:00",
            "violationFrequency": h["violation_count"],
            "dominantVehicle": h["dominant_vehicle"],
            "dominantViolation": h["dominant_violation"],
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    return {
        "total_hotspots": len(results),
        "center": {"lat": center_lat, "lng": center_lng},
        "radius_km": radius_km,
        "hotspots": results
    }

# -----------------------------------------------------
# 2️⃣ HEATMAP — from precomputed points inside JSON
# -----------------------------------------------------
@app.get("/heatmap/violations")
def violation_heatmap(
    center_lat: float = 9.9252,
    center_lng: float = 78.1198,
    radius_km: float = 5.0,
    hour: int = None       # ← now actually used
):
    dists = haversine_km(center_lat, center_lng, HS_LATS, HS_LNGS)
    nearby_idx = np.where(dists <= radius_km)[0]

    points = []
    for i in nearby_idx:
        h = HOTSPOTS[i]
        for pt in h["heatmap_points"]:
            lat, lng = pt[0], pt[1]
            pt_hour = int(pt[2]) if len(pt) > 2 else None

            # If hour filter requested, only include points ±1 hour window
            if hour is not None and pt_hour is not None:
                if abs(pt_hour - hour) > 1:
                    continue

            points.append({"latitude": lat, "longitude": lng})

    return points

# -----------------------------------------------------
# 3️⃣ STATS — aggregated from precomputed data
# -----------------------------------------------------
@app.get("/stats/hourly")
def hourly_stats():
    hour_counts = {str(h): 0 for h in range(24)}
    for h in HOTSPOTS:
        hour_counts[str(h["peak_hour"])] += h["violation_count"]
    return hour_counts

@app.get("/stats/offence")
def get_offence_stats(start_date: str = None, end_date: str = None):
    counts = {}
    for h in HOTSPOTS:
        v = h["dominant_violation"]
        counts[v] = counts.get(v, 0) + h["violation_count"]
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
    return {
        "status": "success",
        "data": [{"violation": k, "count": v} for k, v in sorted_counts]
    }

@app.get("/stats/junctions")
def get_junction_stats(start_date: str = None, end_date: str = None):
    sorted_hotspots = sorted(HOTSPOTS, key=lambda x: x["violation_count"], reverse=True)[:10]
    return {
        "status": "success",
        "data": [
            {
                "junction": f"{h['latitude']:.4f},{h['longitude']:.4f}",
                "count": h["violation_count"]
            }
            for h in sorted_hotspots
        ]
    }