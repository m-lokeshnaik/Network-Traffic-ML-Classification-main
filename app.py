import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'label_encoder' not in st.session_state:
    st.session_state.label_encoder = LabelEncoder()
if 'scaler' not in st.session_state:
    st.session_state.scaler = StandardScaler()
if 'is_model_trained' not in st.session_state:
    st.session_state.is_model_trained = False

# Set page config
st.set_page_config(
    page_title="Network Traffic Classifier",
    page_icon="🌐",
    layout="wide"
)

# Title and description
st.title("Network Traffic Classification")
st.markdown("Upload network traffic data and select a model to classify the traffic patterns.")

# Sidebar for model selection and parameters
st.sidebar.header("Model Configuration")
model_choice = st.sidebar.selectbox(
    "Choose a Classification Model",
    ["Random Forest", "Decision Tree", "Naive Bayes", "K-Nearest Neighbors"]
)

# Model specific parameters
if model_choice == "Random Forest":
    n_estimators = st.sidebar.slider("Number of trees", 50, 500, 250)
    max_depth = st.sidebar.slider("Maximum depth", 10, 100, 50)
elif model_choice == "Decision Tree":
    max_depth_dt = st.sidebar.slider("Maximum depth", 1, 50, 10)
    min_samples_split = st.sidebar.slider("Minimum samples to split", 2, 20, 2)
elif model_choice == "K-Nearest Neighbors":
    n_neighbors = st.sidebar.slider("Number of neighbors", 1, 20, 3)
    weights = st.sidebar.selectbox("Weight function", ["uniform", "distance"])

# Training parameters
st.sidebar.header("Training Parameters")
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)
random_state = st.sidebar.slider("Random State", 0, 100, 42)

# File uploaders
col1, col2 = st.columns(2)
with col1:
    st.subheader("Upload Training Data")
    training_file = st.file_uploader("Upload your training data (CSV)", type="csv", key="training")

with col2:
    st.subheader("Upload Data for Classification")
    prediction_file = st.file_uploader("Upload data to classify (CSV)", type="csv", key="prediction")

if training_file is not None:
    try:
        # Load training data
        train_df = pd.read_csv(training_file)
        st.success("Training file successfully uploaded!")
        
        # Show training data preview
        st.write("Training Data Preview:")
        st.write(train_df.head())
        
        # Select target variable
        st.subheader("Select Target Variable")
        target_column = st.selectbox("Choose the target variable for classification", train_df.columns.tolist())
        
        # Prepare features
        feature_columns = ['Length', 'Protocol', 'Source', 'Destination']
        if not all(col in train_df.columns for col in feature_columns):
            st.error("The training file must contain the following columns: Length, Protocol, Source, Destination")
        else:
            # Preprocess training data
            st.session_state.label_encoder.fit(train_df['Protocol'])
            train_df['Protocol_encoded'] = st.session_state.label_encoder.transform(train_df['Protocol'])
            st.session_state.label_encoder.fit(train_df['Source'])
            train_df['Source_encoded'] = st.session_state.label_encoder.transform(train_df['Source'])
            st.session_state.label_encoder.fit(train_df['Destination'])
            train_df['Destination_encoded'] = st.session_state.label_encoder.transform(train_df['Destination'])
            
            # Create feature matrix
            X = train_df[['Length', 'Protocol_encoded', 'Source_encoded', 'Destination_encoded']]
            y = train_df[target_column]
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            
            # Scale features
            st.session_state.scaler.fit(X_train)
            X_train_scaled = st.session_state.scaler.transform(X_train)
            X_test_scaled = st.session_state.scaler.transform(X_test)
            
            # Initialize model with selected parameters
            if model_choice == "Random Forest":
                st.session_state.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
            elif model_choice == "Decision Tree":
                st.session_state.model = DecisionTreeClassifier(max_depth=max_depth_dt, min_samples_split=min_samples_split, random_state=random_state)
            elif model_choice == "Naive Bayes":
                st.session_state.model = GaussianNB()
            else:  # KNN
                st.session_state.model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
            
            # Train the model
            st.session_state.model.fit(X_train_scaled, y_train)
            st.session_state.is_model_trained = True
            
            # Make predictions on test set
            y_pred = st.session_state.model.predict(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            class_report = classification_report(y_test, y_pred)
            
            # Display training results
            st.success(f"Model training completed! Accuracy: {accuracy:.2%}")
            
            # Display confusion matrix using streamlit
            st.subheader("Confusion Matrix")
            conf_df = pd.DataFrame(
                conf_matrix,
                index=[f'True {i}' for i in np.unique(y)],
                columns=[f'Pred {i}' for i in np.unique(y)]
            )
            st.write(conf_df)
            
            # Display classification report
            st.subheader("Classification Report")
            st.text(class_report)
            
            # If prediction file is uploaded
            if prediction_file is not None:
                try:
                    # Load prediction data
                    pred_df = pd.read_csv(prediction_file)
                    st.write("Data to Classify Preview:")
                    st.write(pred_df.head())
                    
                    # Preprocess prediction data
                    pred_df['Protocol_encoded'] = st.session_state.label_encoder.transform(pred_df['Protocol'])
                    pred_df['Source_encoded'] = st.session_state.label_encoder.transform(pred_df['Source'])
                    pred_df['Destination_encoded'] = st.session_state.label_encoder.transform(pred_df['Destination'])
                    
                    # Create feature matrix for prediction
                    X_pred = pred_df[['Length', 'Protocol_encoded', 'Source_encoded', 'Destination_encoded']]
                    X_pred_scaled = st.session_state.scaler.transform(X_pred)
                    
                    # Add a button to start classification
                    if st.button("Classify Traffic"):
                        try:
                            if not st.session_state.is_model_trained:
                                st.error("Please train the model first by uploading training data.")
                            else:
                                # Make predictions
                                predictions = st.session_state.model.predict(X_pred_scaled)
                                
                                # Add predictions to dataframe
                                pred_df['Predicted_Class'] = predictions
                                
                                # Show results
                                st.subheader("Classification Results")
                                st.write(pred_df[['Length', 'Protocol', 'Source', 'Destination', 'Predicted_Class']].head(10))
                                
                                # Show distribution of predictions using streamlit
                                st.subheader("Distribution of Predictions")
                                pred_counts = pd.Series(predictions).value_counts()
                                st.bar_chart(pred_counts)
                                
                        except Exception as e:
                            st.error(f"An error occurred during classification: {str(e)}")
                            st.error("Please ensure the model is trained and the prediction data format matches the training data.")
                    
                except Exception as e:
                    st.error(f"Error reading prediction file: {str(e)}")
            
    except Exception as e:
        st.error(f"Error reading training file: {str(e)}")

# Add information about the models
st.sidebar.markdown("---")
st.sidebar.subheader("Model Information")
model_info = {
    "Random Forest": "Best for complex patterns and handling large datasets",
    "Decision Tree": "Simple and interpretable, good for understanding decision rules",
    "Naive Bayes": "Fast and efficient, works well with high-dimensional data",
    "K-Nearest Neighbors": "Good for pattern recognition in similar traffic patterns"
}
st.sidebar.write(model_info[model_choice])

# Footer
st.markdown("---")
st.markdown("Created for Network Traffic Classification") 
