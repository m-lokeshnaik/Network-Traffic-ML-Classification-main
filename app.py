import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV, RFE
from numpy import mean, std
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page config
st.set_page_config(page_title="Network Traffic Classifier", page_icon="🌐", layout="wide")

# Title
st.title("🌐 Network Traffic Classification")
st.markdown("Upload your **network traffic CSV file**, choose a classifier, and get results with feature analysis.")

# Sidebar model selection
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
        st.success("✅ File uploaded successfully!")

        st.subheader("📄 Data Preview")
        st.dataframe(df.head())

        required_columns = ['Length', 'Protocol', 'Source', 'Destination']
        if not all(col in df.columns for col in required_columns):
            st.error("❌ The file must contain the following columns: Length, Protocol, Source, Destination")
        else:
            # Encode categorical features
            label_encoders = {}
            for col in ['Protocol', 'Source', 'Destination']:
                le = LabelEncoder()
                df[col + "_encoded"] = le.fit_transform(df[col])
                label_encoders[col] = le

            # Label generation based on quantiles of 'Length'
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
            label_encoder = LabelEncoder()
            df['Label'] = label_encoder.fit_transform(df['Label'])

            # Feature matrix
            features = ['Length', 'Protocol_encoded', 'Source_encoded', 'Destination_encoded']
            X = df[features]
            y = df['Label']

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Model initialization
            if model_choice == "Random Forest":
                model = RandomForestClassifier(n_estimators=250, max_depth=50, random_state=45)
            elif model_choice == "Decision Tree":
                model = DecisionTreeClassifier(random_state=42)
            elif model_choice == "Naive Bayes":
                model = GaussianNB()
            else:
                model = KNeighborsClassifier(n_neighbors=3)

            # Classify Button
            if st.button("🚀 Classify Traffic"):
                with st.spinner("Training and classifying..."):
                    # Train/Test Split
                    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    df['Predicted_Class'] = model.predict(X_scaled)

                    st.success("✅ Classification complete!")

                    # Results
                    st.subheader("🧾 Classification Results")
                    st.dataframe(df[['Length', 'Protocol', 'Source', 'Destination', 'Predicted_Class']].head(10))

                    # Confusion Matrix
                    st.subheader("📊 Confusion Matrix")
                    cm = confusion_matrix(y_test, predictions)
                    labels = ['Low', 'Medium', 'High']
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    st.pyplot(fig)

                    # RFECV Feature Selection
                    st.subheader("🧠 Feature Selection via RFECV")
                    rfecv_model = DecisionTreeClassifier(random_state=42)
                    rfecv = RFECV(estimator=rfecv_model, scoring='accuracy', cv=5, n_jobs=-1)
                    pipeline = Pipeline(steps=[('feature_selection', rfecv), ('classification', rfecv_model)])
                    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1)
                    scores = cross_val_score(pipeline, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
                    st.write(f"**RFECV Accuracy:** {mean(scores):.3f} (±{std(scores):.3f})")

                    # RFE Analysis
                    st.subheader("📌 Feature Ranking via RFE")
                    rfe_selector = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=3)
                    rfe_selector.fit(X_train, y_train)
                    for i, feature in enumerate(features):
                        st.write(f"Feature: {feature:>20} | Selected: {rfe_selector.support_[i]} | Rank: {rfe_selector.ranking_[i]}")

                    # Prediction Distribution
                    st.subheader("📈 Prediction Distribution")
                    st.bar_chart(df['Predicted_Class'].value_counts())

                    # Download results
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name='classified_traffic.csv',
                        mime='text/csv',
                    )

    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")

# Sidebar model information
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Model Info")
model_info = {
    "Random Forest": "Good for complex data patterns and high accuracy.",
    "Decision Tree": "Easy to interpret, fast and decent accuracy.",
    "Naive Bayes": "Best for high-dimensional data, fast and simple.",
    "K-Nearest Neighbors": "Good when similar patterns exist in data."
}
st.sidebar.info(model_info[model_choice])

# Footer
st.markdown("---")
st.markdown("Created with ❤️ for **Network Traffic Classification**")
