# black_hole_dashboard.py (corrected)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

# Load data
df = pd.read_csv("black_hole_data.csv")

st.title("🔭 Black Hole Mass Prediction Dashboard")
st.write("An interactive dashboard to explore black hole properties and predict mass (demo data).")

# Show raw data toggle
if st.checkbox("Show Raw Data"):
    st.dataframe(df.head())

# Filters
galaxy_types = st.multiselect("Select Galaxy Type", df["galaxy_type"].unique(), default=list(df["galaxy_type"].unique()))
spin_min, spin_max = float(df["spin"].min()), float(df["spin"].max())
spin_range = st.slider("Spin Range", spin_min, spin_max, (spin_min, spin_max))
lum_min, lum_max = float(df["luminosity"].min()), float(df["luminosity"].max())
luminosity_range = st.slider("Luminosity Range", lum_min, lum_max, (lum_min, lum_max))

# Apply filters
filtered_df = df.copy()
if galaxy_types:
    filtered_df = filtered_df[filtered_df["galaxy_type"].isin(galaxy_types)]
filtered_df = filtered_df[(filtered_df["spin"] >= spin_range[0]) & (filtered_df["spin"] <= spin_range[1])]
filtered_df = filtered_df[(filtered_df["luminosity"] >= luminosity_range[0]) & (filtered_df["luminosity"] <= luminosity_range[1])]

# Scatter: Luminosity vs Mass
st.subheader("📊 Luminosity vs Mass (scatter)")
fig1, ax1 = plt.subplots(figsize=(6, 4))
sns.scatterplot(data=filtered_df, x="luminosity", y="mass", hue="galaxy_type", s=70, ax=ax1)
ax1.set_xlabel("Luminosity")
ax1.set_ylabel("Mass (solar masses)")
ax1.legend(title="Galaxy Type", bbox_to_anchor=(1.05, 1), loc="upper left")
st.pyplot(fig1)

# Bar: Avg mass by galaxy type
st.subheader("🏷 Avg Mass by Galaxy Type")
avg_mass = filtered_df.groupby("galaxy_type")["mass"].mean().reset_index()
fig2, ax2 = plt.subplots(figsize=(6, 4))
sns.barplot(data=avg_mass, x="galaxy_type", y="mass", ax=ax2)
ax2.set_xlabel("Galaxy Type")
ax2.set_ylabel("Average Mass")
st.pyplot(fig2)

# Histogram: mass distribution
st.subheader("📈 Mass Distribution")
fig3, ax3 = plt.subplots(figsize=(6, 4))
sns.histplot(filtered_df["mass"], bins=15, kde=True, ax=ax3)
ax3.set_xlabel("Mass")
st.pyplot(fig3)

# Prediction input
st.subheader("🔮 Predict Black Hole Mass (demo model)")
luminosity_input = st.number_input("Luminosity", min_value=0.0, value=float(df["luminosity"].median()))
spin_input = st.number_input("Spin", min_value=0.0, value=float(df["spin"].median()))
edd_ratio_input = st.number_input("Eddington Ratio", min_value=0.0, value=float(df["edd_ratio"].median()))

# Try to load a real model if present
model = None
try:
    model = joblib.load("black_hole_model.pkl")
except Exception:
    model = None

if st.button("Predict Mass"):
    if model is not None:
        features = np.array([[luminosity_input, spin_input, edd_ratio_input]])
        prediction = model.predict(features)[0]
        st.success(f"Predicted Black Hole Mass: {prediction:,.0f} solar masses")
    else:
        # fallback demo formula
        prediction = luminosity_input * 0.02 + spin_input * 400000 + edd_ratio_input * 700000
        st.success(f"Predicted Black Hole Mass (demo): {prediction:,.0f} solar masses")

# Insights box
st.subheader("💡 Key Insights (from demo data)")
st.markdown("""
1. Luminosity shows a strong positive relationship with mass.
2. Elliptical galaxies in this sample host larger black holes on average.
3. High spin values in active objects tend to coincide with higher mass.
4. Eddington Ratio helps separate lower-mass (stellar) and higher-mass (supermassive) objects.
""")