"""STEP 6: Redistribution - turn predicted surplus into optimal NGO pickups (min-cost transport LP)."""
import pandas as pd, numpy as np, joblib
from scipy.optimize import linprog

def haversine(lat1, lon1, lat2, lon2):
    p = np.pi/180; a = np.sin((lat2-lat1)*p/2)**2 + np.cos(lat1*p)*np.cos(lat2*p)*np.sin((lon2-lon1)*p/2)**2
    return 12742*np.arcsin(np.sqrt(a))

def predict_surplus(df_feat, date, art):
    day = df_feat[df_feat.date == date].copy()
    X = pd.get_dummies(day.drop(columns=["date", "waste_kg", "lat", "lon"]),
                       columns=["store_id", "store_type", "category"], dtype=int)
    X = X.reindex(columns=art["columns"], fill_value=0)
    day["pred_waste_kg"] = art["model"].predict(X).clip(0)
    return day

def plan(day, stores, ngos, max_km=15, rescue_rate=.8):
    """rescue_rate: share of predicted waste that can realistically be recovered (edible/safe)."""
    # exclude non-perishable-risk irrelevant cats? keep all; cooked meals only to NGOs that accept them
    sup = day.groupby(["store_id", "category"]).pred_waste_kg.sum().mul(rescue_rate).reset_index()
    sup = sup[sup.pred_waste_kg > 5].reset_index(drop=True)
    sup = sup.merge(stores[["store_id", "lat", "lon"]], on="store_id")
    S, N = len(sup), len(ngos)
    dist = np.array([[haversine(s.lat, s.lon, n.lat, n.lon) for n in ngos.itertuples()] for s in sup.itertuples()])
    cost = dist.copy()
    allowed = dist <= max_km
    for i, s in enumerate(sup.itertuples()):                      # cooked food: needs capable NGO & short trip
        for j, n in enumerate(ngos.itertuples()):
            if s.category == "Cooked Meals" and (not n.accepts_cooked or dist[i, j] > 8): allowed[i, j] = False
    c = (cost - 100).flatten()                                    # reward delivering, prefer short distance
    A_sup = np.zeros((S, S*N)); A_cap = np.zeros((N, S*N))
    for i in range(S):
        for j in range(N):
            A_sup[i, i*N+j] = 1; A_cap[j, i*N+j] = 1
    bounds = [(0, None) if allowed[i, j] else (0, 0) for i in range(S) for j in range(N)]
    r = linprog(c, A_ub=np.vstack([A_sup, A_cap]), b_ub=np.r_[sup.pred_waste_kg, ngos.capacity_kg],
                bounds=bounds, method="highs")
    x = r.x.reshape(S, N); rows = []
    for i in range(S):
        for j in range(N):
            if x[i, j] > .5:
                rows.append(dict(store_id=sup.store_id[i], category=sup.category[i], ngo_id=ngos.ngo_id[j],
                                 kg=round(x[i, j], 1), km=round(dist[i, j], 1)))
    out = pd.DataFrame(rows)
    return out, sup.pred_waste_kg.sum()

if __name__ == "__main__":
    art = joblib.load("models/waste_model.joblib")
    df = pd.read_pickle("data/featured.pkl")
    stores, ngos = pd.read_csv("data/stores.csv"), pd.read_csv("data/ngos.csv")
    date = df.date.max()
    day = predict_surplus(df, date, art)
    p, total = plan(day, stores, ngos)
    p.to_csv("reports/pickup_plan.csv", index=False)
    print(f"Date {date.date()} | predicted recoverable surplus: {total:.0f} kg | redistributed: {p.kg.sum():.0f} kg "
          f"({p.kg.sum()/total:.0%}) | meals ≈ {p.kg.sum()/0.4:.0f} | CO2 saved ≈ {p.kg.sum()*2.5:.0f} kg")
    print(p.sort_values("kg", ascending=False).head(10).to_string(index=False))
