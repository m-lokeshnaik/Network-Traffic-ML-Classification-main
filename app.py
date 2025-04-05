import streamlit as st
 import pandas as pd
 import numpy as np
 from sklearn.preprocessing import LabelEncoder, StandardScaler
 from sklearn.naive_bayes import GaussianNB
 from sklearn.tree import DecisionTreeClassifier
 from sklearn.ensemble import RandomForestClassifier
 from sklearn.neighbors import KNeighborsClassifier
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
     # Load and preprocess data
     try:
         df = pd.read_csv(uploaded_file)
         st.success("File successfully uploaded!")
         
         # Show data preview
         st.subheader("Data Preview")
         st.write(df.head())
         
         # Prepare features
         required_columns = ['Length', 'Protocol', 'Source', 'Destination']
         if not all(col in df.columns for col in required_columns):
             st.error("The uploaded file must contain the following columns: Length, Protocol, Source, Destination")
         else:
             # Preprocess data
             label_encoder = LabelEncoder()
             df['Protocol_encoded'] = label_encoder.fit_transform(df['Protocol'])
             df['Source_encoded'] = label_encoder.fit_transform(df['Source'])
             df['Destination_encoded'] = label_encoder.fit_transform(df['Destination'])
             
             # Create feature matrix
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
             
             # Add a button to start classification
             if st.button("Classify Traffic"):
                 try:
                     # Make predictions
                     predictions = model.predict(X_scaled)
                     
                     # Add predictions to dataframe
                     df['Predicted_Class'] = predictions
                     
                     # Show results
                     st.subheader("Classification Results")
                     st.write(df[['Length', 'Protocol', 'Source', 'Destination', 'Predicted_Class']].head(10))
                     
                     # Show distribution of predictions
                     st.subheader("Distribution of Predictions")
                     st.bar_chart(df['Predicted_Class'].value_counts())
                     
                 except Exception as e:
                     st.error(f"An error occurred during classification: {str(e)}")
                 
     except Exception as e:
         st.error(f"Error reading the file: {str(e)}")
 
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
 
 # Add requirements section
 requirements = """
 Requirements:
 - Python 3.7+
 - streamlit
 - pandas
 - numpy
 - scikit-learn
 """
 
 # Footer
 st.markdown("---")
 st.markdown("Created for Network Traffic Classification") 
