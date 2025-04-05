import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Set page config
st.set_page_config(
    page_title="Network Traffic Classifier",
    page_icon="🌐",
    layout="wide"
)

# Title and description
st.title("Network Traffic Classification")
st.markdown("Upload network traffic data and select a model to classify the traffic patterns.")

# Sidebar for model selection
st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose a Classification Model",
    ["Random Forest", "Decision Tree", "Naive Bayes", "K-Nearest Neighbors"]
)

# File uploader
uploaded_file = st.file_uploader("Upload your network traffic data (CSV)", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File successfully uploaded!")
        
        # Show data preview
        st.subheader("Data Preview")
        st.write(df.head())

        # Check required columns
        required_columns = ['Length', 'Protocol', 'Source', 'Destination']
        if not all(col in df.columns for col in required_columns):
            st.error("The uploaded file must contain the following columns: Length, Protocol, Source, Destination")
        else:
            # Encode categorical features
            label_encoders = {}
            for col in ['Source', 'Destination', 'Protocol']:
                le = LabelEncoder()
                df[col + '_encoded'] = le.fit_transform(df[col])
                label_encoders[col] = le

            # Categorize 'Length' into Low, Medium, High
            q1 = df['Length'].quantile(0.33)
            q2 = df['Length'].quantile(0.66)

            def categorize_length(length):
                if length <= q1:
                    return 'Low'
                elif length <= q2:
                    return 'Medium'
                else:
                    return 'High'

            df['Label'] = df['Length'].apply(categorize_length)
            df['Label'] = LabelEncoder().fit_transform(df['Label'])  # 0: Low, 1: Medium, 2: High

            # Feature matrix
            X = df[['Length', 'Protocol_encoded', 'Source_encoded', 'Destination_encoded']]
            y = df['Label']

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Initialize selected model
            if model_choice == "Random Forest":
                model = RandomForestClassifier(n_estimators=250, max_depth=50, random_state=45)
            elif model_choice == "Decision Tree":
                model = DecisionTreeClassifier(random_state=42)
            elif model_choice == "Naive Bayes":
                model = GaussianNB()
            else:  # KNN
                model = KNeighborsClassifier(n_neighbors=3)

            # Classify traffic
            if st.button("Classify Traffic"):
                with st.spinner("Processing and classifying..."):
                    # Split data for evaluation
                    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)

                    # Evaluation
                    st.subheader("Model Evaluation")
                    st.write("**Accuracy:**", accuracy_score(y_test, predictions))
                    st.text("Classification Report:")
                    st.text(classification_report(y_test, predictions))

                    # Predict on all data and show results
                    df['Predicted_Class'] = model.predict(X_scaled)
                    st.subheader("Classification Results (Top 10 Rows)")
                    st.write(df[['Length', 'Protocol', 'Source', 'Destination', 'Predicted_Class']].head(10))

                    # Prediction distribution
                    st.subheader("Distribution of Predictions")
                    st.bar_chart(df['Predicted_Class'].value_counts())

                    # Download button
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name='classified_traffic.csv',
                        mime='text/csv',
                    )

    except Exception as e:
        st.error(f"Error reading the file: {str(e)}")

# Sidebar model info
st.sidebar.markdown("---")
st.sidebar.subheader("Model Information")
model_info = {
    "Random Forest": "Best for complex patterns and handling large datasets",
    "Decision Tree": "Simple and interpretable, good for understanding decision rules",
    "Naive Bayes": "Fast and efficient, works well with high-dimensional data",
    "K-Nearest Neighbors": "Good for pattern recognition in similar traffic patterns"
}
st.sidebar.write(model_info[model_choice])

# Requirements
st.markdown("---")
st.markdown("**Requirements:**\n- Python 3.7+\n- streamlit\n- pandas\n- numpy\n- scikit-learn")
st.markdown("Created for Network Traffic Classification 🚀")
