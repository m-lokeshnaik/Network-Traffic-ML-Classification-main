import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Set page config
st.set_page_config(
    page_title="Network Traffic Classifier",
    page_icon="🌐",
    layout="wide"
)

# Title and description
st.title("🌐 Network Traffic Classification")
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
        st.success("✅ File successfully uploaded!")

        # Show data preview
        st.subheader("📄 Data Preview")
        st.write(df.head())

        # Check required columns
        required_columns = ['Length', 'Protocol', 'Source', 'Destination']
        if not all(col in df.columns for col in required_columns):
            st.error("❌ The uploaded file must contain the following columns: Length, Protocol, Source, Destination")
        else:
            # Encode categorical features
            protocol_encoder = LabelEncoder()
            source_encoder = LabelEncoder()
            dest_encoder = LabelEncoder()

            df['Protocol_encoded'] = protocol_encoder.fit_transform(df['Protocol'])
            df['Source_encoded'] = source_encoder.fit_transform(df['Source'])
            df['Destination_encoded'] = dest_encoder.fit_transform(df['Destination'])

            # Feature matrix
            X = df[['Length', 'Protocol_encoded', 'Source_encoded', 'Destination_encoded']]

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

            # Train and predict
            if st.button("🚦 Classify Traffic"):
                with st.spinner("🔍 Processing and classifying..."):
                    if 'Label' in df.columns:
                        y = df['Label']
                        model.fit(X_scaled, y)
                        predictions = model.predict(X_scaled)
                        df['Predicted_Class'] = predictions

                        # Evaluation Metrics
                        accuracy = accuracy_score(y, predictions)
                        precision = precision_score(y, predictions, average='weighted', zero_division=0)
                        recall = recall_score(y, predictions, average='weighted', zero_division=0)
                        f1 = f1_score(y, predictions, average='weighted', zero_division=0)

                        st.success("✅ Classification complete using training data!")
                        st.subheader("📊 Model Performance Metrics")
                        st.markdown(f"- **Accuracy**: `{accuracy:.2f}`")
                        st.markdown(f"- **Precision**: `{precision:.2f}`")
                        st.markdown(f"- **Recall**: `{recall:.2f}`")
                        st.markdown(f"- **F1 Score**: `{f1:.2f}`")

                        # Confusion Matrix
                        st.subheader("🧮 Confusion Matrix")
                        labels = sorted(df['Label'].unique())
                        cm = confusion_matrix(y, predictions, labels=labels)
                        fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                    xticklabels=labels, yticklabels=labels, ax=ax)
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        st.pyplot(fig)

                    else:
                        st.warning("⚠️ No 'Label' column found. Model will predict without training. Results may be invalid.")
                        try:
                            predictions = model.predict(X_scaled)
                            df['Predicted_Class'] = predictions
                        except Exception as e:
                            st.error(f"❌ Model couldn't classify without training: {e}")
                            st.stop()

                    # Display results
                    st.subheader("📋 Classification Results")
                    st.write(df[['Length', 'Protocol', 'Source', 'Destination', 'Predicted_Class']].head(10))

                    # Prediction distribution
                    st.subheader("📈 Distribution of Predictions")
                    st.bar_chart(df['Predicted_Class'].value_counts())

                    # Download button
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name='classified_traffic.csv',
                        mime='text/csv',
                    )

    except Exception as e:
        st.error(f"❌ Error reading the file: {str(e)}")

# Sidebar model info
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Model Information")
model_info = {
    "Random Forest": "Best for complex patterns and handling large datasets.",
    "Decision Tree": "Simple and interpretable, good for understanding decision rules.",
    "Naive Bayes": "Fast and efficient, works well with high-dimensional data.",
    "K-Nearest Neighbors": "Good for pattern recognition in similar traffic patterns."
}
st.sidebar.write(model_info[model_choice])

# Footer
st.markdown("---")
st.markdown("✅ **Requirements**: Python 3.7+, streamlit, pandas, numpy, scikit-learn, seaborn, matplotlib")
st.markdown("🔧 Created for Network Traffic Classification 🚀")
