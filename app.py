import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Load your trained model pipeline
@st.cache_resource
def load_model():
    # You can also replace this with loading from disk if saved earlier using joblib
    df = pd.read_csv("rough_pricing.csv", encoding='latin1')
    
    features = [
        'Weight', 'Color', 'Clarity', 'Cut', 'Polish', 'Symmetry',
        'Table %', 'Depth %', 'Shape', 'Lab', 'Eye Clean'
    ]
    
    # Clean data
    for col in features:
        if df[col].dtype == 'object':
            df[col].fillna('Unknown', inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

    df['Table_Ideal'] = df['Table %'].between(58, 62).astype(int)
    df['Depth_Ideal'] = df['Depth %'].between(60, 62.5).astype(int)

    features += ['Table_Ideal', 'Depth_Ideal']

    X = df[features]
    y = df['Rapnet Discount %']

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    pipeline.fit(X, y)

    return pipeline, features

model, required_features = load_model()

st.title("💎 Diamond Discount Predictor")
st.write("Upload a CSV file to get predicted RapNet Discounts for your diamonds.")

uploaded_file = st.file_uploader("Upload your diamond dataset (CSV)", type=["csv"])

if uploaded_file:
    input_df = pd.read_csv(uploaded_file)

    # Feature engineering
    if 'Table %' in input_df.columns:
        input_df['Table_Ideal'] = input_df['Table %'].between(58, 62).astype(int)
    if 'Depth %' in input_df.columns:
        input_df['Depth_Ideal'] = input_df['Depth %'].between(60, 62.5).astype(int)

    # Fill missing features if any
    for feature in required_features:
        if feature not in input_df.columns:
            if feature in ['Table_Ideal', 'Depth_Ideal']:
                input_df[feature] = 1
            elif feature in ['Weight', 'Table %', 'Depth %']:
                input_df[feature] = 60.0
            else:
                input_df[feature] = 'Unknown'

    input_final = input_df[required_features]
    predictions = model.predict(input_final)
    input_df['Predicted Discount (%)'] = predictions

    st.success("✅ Prediction complete!")
    st.dataframe(input_df)

    csv = input_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions as CSV", data=csv, file_name='predicted_discounts.csv', mime='text/csv')
