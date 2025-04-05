# Network Traffic Classification App

This Streamlit application allows you to classify network traffic patterns using different machine learning models.

## Features

- Multiple model selection (Random Forest, Decision Tree, Naive Bayes, KNN)
- Interactive data upload and preview
- Real-time classification
- Visualization of results
- Model information and descriptions

## Requirements

- Python 3.7+
- Required packages listed in `requirements.txt`

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd network-traffic-classification
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided URL (typically http://localhost:8501)

3. Upload your network traffic data (CSV file)
   - Required columns: Length, Protocol, Source, Destination

4. Select a classification model from the sidebar

5. Click "Classify Traffic" to see the results

## Data Format

Your CSV file should contain the following columns:
- Length: Packet length
- Protocol: Network protocol
- Source: Source IP/address
- Destination: Destination IP/address

## Models Available

1. Random Forest
   - Best for complex patterns and handling large datasets
   - High accuracy and good generalization

2. Decision Tree
   - Simple and interpretable
   - Good for understanding decision rules

3. Naive Bayes
   - Fast and efficient
   - Works well with high-dimensional data

4. K-Nearest Neighbors
   - Good for pattern recognition
   - Works well with similar traffic patterns

## License

This project is licensed under the MIT License - see the LICENSE file for details. 