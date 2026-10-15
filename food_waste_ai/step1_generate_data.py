"""STEP 1 - DATA: builds a realistic synthetic retail food-waste dataset.
Replace with real data (e.g. Kaggle 'Food Waste' / supermarket sales) keeping the same columns."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)

stores = pd.DataFrame({
    "store_id": [f"S{i:02d}" for i in range(1, 16)],
    "store_type": rng.choice(["Supermarket", "Restaurant", "Bakery", "Hotel"], 15, p=[.4, .25, .2, .15]),
    "lat": 16.30 + rng.uniform(0, .15, 15), "lon": 80.40 + rng.uniform(0, .15, 15),
    "size_factor": rng.uniform(.6, 1.8, 15)})
cats = pd.DataFrame({
    "category": ["Bakery", "Dairy", "Vegetables", "Fruits", "Cooked Meals", "Meat", "Packaged", "Beverages"],
    "shelf_life_days": [2, 7, 5, 6, 1, 3, 90, 120],
    "base_demand_kg": [40, 55, 70, 60, 50, 35, 80, 90],
    "overstock_bias": [.18, .12, .20, .18, .25, .12, .05, .04]})

dates = pd.date_range("2023-01-01", "2024-12-31")
d = pd.DataFrame({"date": dates})
d["dow"] = d.date.dt.dayofweek; d["month"] = d.date.dt.month
d["is_weekend"] = (d.dow >= 5).astype(int)
d["temperature"] = 28 + 8*np.sin((d.month-3)/12*2*np.pi) + rng.normal(0, 2, len(d))
d["rain_mm"] = np.where(rng.random(len(d)) < .25, rng.gamma(2, 6, len(d)), 0)
festivals = pd.to_datetime(["2023-01-14","2023-03-08","2023-08-15","2023-10-24","2023-11-12","2023-12-25",
                            "2024-01-14","2024-03-25","2024-08-15","2024-10-12","2024-11-01","2024-12-25"])
d["is_holiday"] = d.date.isin(festivals).astype(int)

df = d.merge(stores, how="cross").merge(cats, how="cross")
n = len(df)
df["promo"] = (rng.random(n) < .12).astype(int)
season = 1 + .15*df.is_weekend + .25*df.is_holiday - .12*(df.rain_mm > 10) + .05*(df.temperature > 33)
df["footfall"] = (300*df.size_factor*season*rng.normal(1, .08, n)).round()
demand = df.base_demand_kg*df.size_factor*season*(1+.2*df.promo)*rng.normal(1, .12, n)
# cold-weather / rain lowers fresh demand, heat raises beverage demand
demand *= np.where((df.category == "Beverages") & (df.temperature > 33), 1.2, 1)
df["demand_kg"] = demand.clip(1).round(1)
# stock decisions: manager's estimate with overstocking bias (perishables overstocked more)
df["stocked_kg"] = (df.base_demand_kg*df.size_factor*(1+df.overstock_bias)*rng.normal(1, .15, n)*
                    (1+.1*df.is_holiday)).clip(1).round(1)
sold = np.minimum(df.stocked_kg, df.demand_kg)
leftover = df.stocked_kg - sold
# fraction of leftover that actually spoils depends on shelf life
spoil_frac = np.clip(1.2/np.sqrt(df.shelf_life_days), .05, 1)
df["waste_kg"] = (leftover*spoil_frac).round(2)
df["sold_kg"] = sold.round(1)
df = df.drop(columns=["overstock_bias", "base_demand_kg", "demand_kg", "sold_kg"])  # hide leakage columns
df.to_csv("data/food_waste.csv", index=False)
stores.to_csv("data/stores.csv", index=False)

# NGOs / food banks
ngos = pd.DataFrame({
    "ngo_id": [f"N{i}" for i in range(1, 9)],
    "lat": 16.30 + rng.uniform(0, .15, 8), "lon": 80.40 + rng.uniform(0, .15, 8),
    "capacity_kg": rng.integers(80, 220, 8),
    "accepts_cooked": rng.choice([0, 1], 8, p=[.5, .5])})
ngos.to_csv("data/ngos.csv", index=False)
print("Rows:", len(df), "| saved data/food_waste.csv, stores.csv, ngos.csv")
