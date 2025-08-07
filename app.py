import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 💎 1. Add the custom styling function here
def apply_custom_style():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1585238341986-2b61f1a1bbaa?ixlib=rb-4.0.3&auto=format&fit=crop&w=1950&q=80");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            font-family: 'Arial', sans-serif;
            color: #ffffff;
        }

        h1 {
            color: #ffd700;
            text-align: center;
        }

        .stFileUploader, .stButton {
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            padding: 10px;
        }

        .stDataFrame {
            background-color: rgba(255, 255, 255, 0.95);
            color: black;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

# 💎 2. Call the function before UI
apply_custom_style()



@st.cache_data
def load_model():
    df = pd.read_csv("rough_pricing.csv", encoding='latin1', low_memory=False)

    # Drop rows with missing target values
    df = df.dropna(subset=["Rapnet Discount %"])

    selected_features = [
        "Weight", "Color", "Clarity", "Cut", "Polish", "Symmetry",
        "Table %", "Depth %", "Shape", "Lab", "Eye Clean"
    ]

    # Feature engineering
    df['Table_Ideal'] = df['Table %'].between(58, 62).astype(int)
    df['Depth_Ideal'] = df['Depth %'].between(60, 62.5).astype(int)
    selected_features += ['Table_Ideal', 'Depth_Ideal']

    X = df[selected_features]
    y = df["Rapnet Discount %"]

    # Preprocessing
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=50, random_state=42))
    ])

    # Train on subset for faster performance
    X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.2, random_state=42)
    model.fit(X_sample, y_sample)

    return model, selected_features

# Load model and features
model, required_features = load_model()

st.title("💎 Diamond Pricing Calculator")

uploaded_file = st.file_uploader("Upload CSV file with diamond data", type=["csv"])

if uploaded_file:
    try:
        input_df = pd.read_csv(uploaded_file, encoding='latin1')

        # Strip column names of spaces and fix naming issues
        input_df.columns = input_df.columns.str.strip()
        rename_map = {
            col: 'Table %' for col in input_df.columns if col.lower().replace(" ", "") == 'table%'
        }
        rename_map.update({
            col: 'Depth %' for col in input_df.columns if col.lower().replace(" ", "") == 'depth%'
        })
        input_df.rename(columns=rename_map, inplace=True)

        # Feature engineering on uploaded file
        input_df['Table_Ideal'] = input_df['Table %'].between(58, 62).astype(int)
        input_df['Depth_Ideal'] = input_df['Depth %'].between(60, 62.5).astype(int)

        # Fill missing required columns
        for feature in required_features:
            if feature not in input_df.columns:
                if feature in ['Table_Ideal', 'Depth_Ideal']:
                    input_df[feature] = 1
                elif feature in ['Weight', 'Table %', 'Depth %']:
                    input_df[feature] = 60.0
                else:
                    input_df[feature] = 'Unknown'

        # Reorder columns
        input_df = input_df[required_features]

        # Predict
        predictions = model.predict(input_df)
        input_df["Predicted Discount %"] = predictions

        # Display
        st.success("✅ Predictions completed!")
        st.dataframe(input_df)

        # Download button
        csv = input_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions as CSV", data=csv, file_name="diamond_predictions.csv", mime='text/csv')

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
else:
    st.info("Please upload a CSV file to get started.")
