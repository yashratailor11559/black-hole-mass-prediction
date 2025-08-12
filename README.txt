# Black Hole Mass Prediction Dashboard

## Overview
This Streamlit dashboard allows users to:
- Explore a dataset of galaxies with properties such as spin, luminosity, Eddington ratio, and black hole mass.
- View visualizations such as scatter plots, bar plots, and histograms.
- Filter data interactively and (if a model is provided) predict black hole mass.

## Files
- `black_hole_data.csv` — sample dataset.
- `black_hole_dashboard.py` — Streamlit app file.
- `Black_Hole_Prediction_Presentation.pptx` — presentation slides.

## Running the Dashboard
1. Install required packages:
   ```bash
   pip install streamlit pandas matplotlib seaborn joblib
   ```
2. Run the dashboard:
   ```bash
   streamlit run black_hole_dashboard.py
   ```

## Model Integration
To enable predictions:
1. Train a regression model using your dataset.
2. Save it as `black_hole_model.pkl`:
   ```python
   import joblib
   joblib.dump(model, "black_hole_model.pkl")
   ```
3. Place the `.pkl` file in the same directory as `black_hole_dashboard.py`.
4. The app will automatically load it and allow users to enter spin, luminosity, and Eddington ratio to predict mass.

## Author
Yashra Tailor
