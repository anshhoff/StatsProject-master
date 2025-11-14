# Student Placement Predictor

A binary classification project that predicts whether a student will be placed or not based on their CGPA and number of internships.

## Project Structure

```
harshit/
├── Placement.csv                      # Dataset (1000 students)
├── logistic_regression_model.py       # Model training script
├── models.json                        # Exported model coefficients
├── index.html                         # Standalone web application
├── logistic_regression_results.png    # Visualization results
└── venv/                              # Python virtual environment
```

## Features

- **Binary Classification**: Predicts Placed (1) or Not Placed (0)
- **Features Used**:
  - CGPA (0-10 scale)
  - Number of Internships
  - CGPA_squared (engineered feature)
  - Internships_squared (engineered feature)
  - CGPA × Internships (interaction feature)

## Model

- **Algorithm**: Logistic Regression
- **Preprocessing**: StandardScaler for feature normalization
- **Class Balancing**: `class_weight='balanced'` to handle imbalanced data
- **Evaluation Metrics**:
  - Accuracy
  - ROC-AUC Score
  - Precision, Recall, F1-Score
  - Confusion Matrix

## Web Application

The project includes a **standalone frontend-only** web application that runs entirely in the browser without any backend server.

### Features:
- ✅ No backend required - all predictions run in JavaScript
- ✅ Real-time predictions using logistic regression coefficients
- ✅ Interactive chart showing placement probability
- ✅ Color-coded results (green for placed, red for not placed)
- ✅ Responsive design

### How to Run:

1. **Start a simple HTTP server**:
   ```bash
   python -m http.server 8000
   ```

2. **Open in browser**:
   ```
   http://localhost:8000/index.html
   ```

3. **Enter your details**:
   - CGPA (0-10)
   - Number of Internships (0-10)
   - Click "Predict Placement"

## Training the Model

To retrain the model with new data:

```bash
# Activate virtual environment
source venv/bin/activate

# Run the training script
python logistic_regression_model.py
```

This will:
1. Train the logistic regression model
2. Export coefficients to `models.json`
3. Generate visualizations in `logistic_regression_results.png`
4. Print performance metrics

## Model Performance

- **Test Accuracy**: ~51%
- **ROC-AUC Score**: ~0.53
- **Note**: The relatively low accuracy suggests that CGPA and Internships alone may not be sufficient predictors. Consider adding more features for better performance.

## Technologies Used

- **Python**: scikit-learn, pandas, numpy, matplotlib, seaborn
- **Frontend**: HTML, CSS, JavaScript
- **Visualization**: Chart.js
- **Model Format**: JSON (coefficients and scaler parameters)

## How It Works

### Training (Python):
1. Load and preprocess data
2. Engineer polynomial and interaction features
3. Split data (80% train, 20% test)
4. Standardize features using StandardScaler
5. Train logistic regression classifier
6. Export model coefficients to JSON

### Prediction (JavaScript):
1. Load model coefficients from `models.json`
2. Prepare features (add squared and interaction terms)
3. Standardize using saved scaler parameters
4. Calculate: z = w^T × x + b
5. Apply sigmoid: probability = 1 / (1 + e^(-z))
6. Predict: Placed if probability ≥ 0.5, else Not Placed
7. Display results with visualization

## Dataset

- **Source**: Placement.csv
- **Size**: 1000 students
- **Features**:
  - CGPA (5.34 - 10.58)
  - Internships (0 - 4)
  - Placed (Yes/No)
  - Salary (INR LPA) - Not used in this binary classification model

## License

This project is for educational purposes.
