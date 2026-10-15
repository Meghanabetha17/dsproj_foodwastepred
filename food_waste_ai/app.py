"""STEP 7: Dashboard.  Run:  streamlit run app.py"""
import streamlit as st, pandas as pd, joblib
from step3_redistribute import predict_surplus, plan
st.set_page_config(page_title="Food Waste AI", layout="wide")
st.title("🍲 AI Food Waste Prediction & Redistribution")
art = joblib.load("models/waste_model.joblib"); df = pd.read_pickle("data/featured.pkl")
stores, ngos = pd.read_csv("data/stores.csv"), pd.read_csv("data/ngos.csv")
dates = sorted(df.date.unique())[-30:]
date = st.sidebar.selectbox("Date", dates, index=len(dates)-1)
rate = st.sidebar.slider("Recoverable share of waste", .3, 1.0, .8)
km = st.sidebar.slider("Max pickup distance (km)", 3, 30, 15)
day = predict_surplus(df, pd.Timestamp(date), art)
p, total = plan(day, stores, ngos, km, rate)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Predicted waste (kg)", f"{day.pred_waste_kg.sum():.0f}")
c2.metric("Redistributed (kg)", f"{p.kg.sum():.0f}")
c3.metric("Meals provided", f"{p.kg.sum()/0.4:.0f}")
c4.metric("CO₂ avoided (kg)", f"{p.kg.sum()*2.5:.0f}")
st.subheader("Predicted vs actual waste by category")
st.bar_chart(day.groupby("category")[["pred_waste_kg", "waste_kg"]].sum())
st.subheader("Pickup plan"); st.dataframe(p, use_container_width=True)
st.map(pd.concat([stores[["lat", "lon"]], ngos[["lat", "lon"]]]))
