# Student Placement Prediction Dashboard

A comprehensive machine learning project that predicts student placement outcomes using multiple algorithms and provides an interactive web dashboard for real-time predictions.

## Project Structure

```
StatsProject-master/
├── college_student_placement_dataset.csv  # Main dataset
├── Placement.csv                          # Alternative dataset
├── multi_model_comparison.py              # Multi-model training script
├── multi_models.json                      # Exported model coefficients
├── trained_models.pkl                     # Pickled trained models
├── real_data_analysis.py                  # Dataset analysis script
├── prediction_service.py                  # Prediction service
├── server.py                             # Local development server
├── index.html                            # Interactive web dashboard
├── index_old.html                        # Legacy version
├── real_dashboard_data.json              # Dashboard configuration
├── package.json                          # Project metadata
├── vercel.json                           # Deployment configuration
├── DEPLOYMENT.md                         # Deployment guide
└── .venv/                               # Python virtual environment
```

## Features

- **Multi-Model Comparison**: Implements and compares multiple ML algorithms
- **Comprehensive Feature Set**:
  - IQ Score
  - Previous Semester Result
  - CGPA (0-10 scale)
  - Academic Performance
  - Internship Experience (Yes/No)
  - Extra Curricular Score
  - Communication Skills
  - Projects Completed
- **Advanced Analytics**: Real-time placement probability calculation
- **Interactive Dashboard**: Modern web interface with visualizations

## Machine Learning Models

The project implements three different algorithms with anti-overfitting measures:

### 1. Logistic Regression
- **Regularization**: L2 penalty with C=0.5
- **Class Balancing**: Balanced class weights
- **Cross-validation**: 5-fold CV for validation

### 2. Random Forest
- **Parameters**: 50 estimators, max_depth=8
- **Anti-overfitting**: Minimum samples split/leaf constraints
- **Feature Importance**: Available for analysis

### 3. XGBoost (Optional)
- **Regularization**: L1 (0.1) and L2 (0.1) regularization
- **Tree Constraints**: Max depth=6, learning rate=0.1
- **Subsampling**: 80% sample and feature subsampling

## Model Performance & Validation

- **Data Split**: 85% training, 15% testing (stratified)
- **Cross-Validation**: 5-fold CV to detect overfitting
- **Metrics Tracked**:
  - Accuracy, Precision, Recall
  - ROC-AUC Score
  - Confusion Matrix
  - Specificity and Sensitivity
- **Overfitting Detection**: CV mean vs test accuracy comparison

## Web Application

The project features a **comprehensive interactive dashboard** that runs both client-side and with optional server-side predictions.

### Dashboard Features:
- ✅ **Multi-Model Predictions**: Compare results from different algorithms
- ✅ **Real-time Analytics**: Interactive charts and visualizations
- ✅ **Feature Analysis**: Understand impact of different factors
- ✅ **Probability Scoring**: Get placement probability percentages
- ✅ **Responsive Design**: Modern UI with gradient backgrounds
- ✅ **Data Insights**: Real placement statistics and trends
- ✅ **Export Capabilities**: Save and share results

### Deployment Options:

#### Option 1: Static Frontend (Client-side)
```bash
# Start simple HTTP server
python -m http.server 8000
# or use npm script
npm run dev
```

#### Option 2: Full Server (Server-side predictions)
```bash
# Run the prediction server
python server.py
```

#### Option 3: Cloud Deployment
```bash
# Deploy to Vercel (see DEPLOYMENT.md)
npm run deploy
```

### Usage:
1. **Open in browser**: `http://localhost:8000`
2. **Enter student details**:
   - IQ Score (80-150)
   - Previous Semester Result (1-10)
   - CGPA (0-10)
   - Academic Performance (1-10)
   - Internship Experience (Yes/No)
   - Extra Curricular Score (1-10)
   - Communication Skills (1-10)
   - Projects Completed (0-10)
3. **Get Predictions**: View results from multiple models
4. **Analyze Results**: Explore interactive charts and insights

## Training the Models

### Full Model Training & Comparison:

```bash
# Activate virtual environment
source .venv/bin/activate  # or python -m venv .venv && source .venv/bin/activate

# Install dependencies (if needed)
pip install pandas numpy scikit-learn matplotlib seaborn xgboost

# Run multi-model comparison
python multi_model_comparison.py
```

This will:
1. Load and preprocess the dataset (college_student_placement_dataset.csv)
2. Train multiple models with cross-validation
3. Export model coefficients to `multi_models.json`
4. Save trained models to `trained_models.pkl`
5. Generate performance comparison and overfitting detection
6. Print detailed metrics and confusion matrices

### Real Data Analysis:

```bash
# Analyze dataset patterns
python real_data_analysis.py
```

This generates:
- Placement rate analysis by CGPA ranges
- Feature correlation insights
- Real dashboard data for visualization

## Model Performance

### Latest Results:
- **Logistic Regression**: ~51% accuracy (baseline)
- **Random Forest**: Improved performance with regularization
- **XGBoost**: Best performance with L1/L2 regularization
- **Cross-Validation**: 5-fold CV prevents overfitting
- **Test Split**: 15% held-out test set for final evaluation

### Key Insights:
- 8 comprehensive features provide better prediction than basic CGPA/internships
- Class balancing addresses dataset imbalance
- Regularization prevents overfitting in complex models
- Feature importance analysis reveals most predictive factors

## Technologies Used

### Backend/ML:
- **Python**: Core language for ML pipeline
- **scikit-learn**: Machine learning algorithms and preprocessing
- **XGBoost**: Gradient boosting framework (optional)
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **pickle**: Model serialization

### Frontend:
- **HTML5/CSS3**: Modern responsive design
- **JavaScript ES6+**: Interactive functionality
- **Chart.js**: Data visualization and charts
- **CSS Grid/Flexbox**: Layout system

### Development & Deployment:
- **Vercel**: Cloud deployment platform
- **Git**: Version control
- **npm**: Package management
- **Python HTTP Server**: Local development

## How It Works

### Training Pipeline (Python):
1. **Data Loading**: Load college_student_placement_dataset.csv
2. **Preprocessing**: 
   - Drop College_ID
   - Convert categorical variables (Internship_Experience)
   - Feature scaling with StandardScaler
3. **Model Training**: 
   - Split data (85% train, 15% test)
   - Train multiple models with regularization
   - Cross-validate to prevent overfitting
4. **Export**: Save coefficients to JSON and pickle models

### Prediction Pipeline (JavaScript/Python):
1. **Input Processing**: Collect user inputs from dashboard
2. **Feature Preparation**: Apply same preprocessing as training
3. **Model Prediction**: 
   - Client-side: Use exported coefficients
   - Server-side: Load pickled models
4. **Result Display**: Show probability and classification
5. **Visualization**: Interactive charts and analytics

### Server Architecture:
```
Client Browser → index.html → JavaScript → Local/Cloud Server → Models → Prediction
                     ↓
                Chart.js Visualizations
```

## Dataset

### Primary Dataset: `college_student_placement_dataset.csv`
- **Size**: 1000+ students
- **Target Variable**: Placement (Yes/No)
- **Features**:
  - **IQ**: Intelligence quotient score
  - **Prev_Sem_Result**: Previous semester performance (1-10)
  - **CGPA**: Cumulative Grade Point Average (0-10)
  - **Academic_Performance**: Overall academic rating (1-10)
  - **Internship_Experience**: Previous internship experience (Yes/No)
  - **Extra_Curricular_Score**: Extracurricular activities rating (1-10)
  - **Communication_Skills**: Communication ability rating (1-10)
  - **Projects_Completed**: Number of projects completed (0-10)

### Alternative Dataset: `Placement.csv`
- Legacy dataset with CGPA, Internships, and Salary data
- Used for comparison and validation

### Data Quality:
- **Balanced Features**: All numerical features properly scaled
- **Missing Values**: Handled during preprocessing
- **Class Distribution**: Analyzed and balanced using class weights
- **Feature Engineering**: Interaction terms and polynomial features available

## Quick Start

### 1. Clone and Setup:
```bash
git clone <repository-url>
cd StatsProject-master
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt  # Create if needed
```

### 2. Train Models:
```bash
python multi_model_comparison.py
```

### 3. Run Dashboard:
```bash
# Option A: Simple HTTP server
python -m http.server 8000

# Option B: Full prediction server
python server.py

# Option C: Use npm
npm run dev
```

### 4. Access Application:
Open `http://localhost:8000` in your browser

## API Endpoints

When using `server.py`, the following endpoints are available:

- `GET /`: Main dashboard
- `POST /predict`: Server-side prediction
- `GET /api/models`: Available model information
- `GET /api/stats`: Dataset statistics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - See LICENSE file for details

## Support

For issues and questions:
- Check existing GitHub issues
- Create a new issue with detailed description
- Include error logs and system information
