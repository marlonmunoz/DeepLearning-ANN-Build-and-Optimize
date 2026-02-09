# DeepLearning-ANN-Build-and-Optimize

## Artificial Neural Networks: Build & Optimize

A comprehensive deep learning project for customer churn prediction using Artificial Neural Networks (ANN) built with TensorFlow and Keras. This project demonstrates end-to-end machine learning workflow including data preprocessing, model architecture design, training optimization, and performance evaluation.

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Project Workflow](#project-workflow)
- [Model Architecture](#model-architecture)
- [Performance Metrics](#performance-metrics)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Key Learnings](#key-learnings)

---

## 🎯 Project Overview

This project builds a neural network to predict customer churn for a bank. The model analyzes customer data (demographics, account information, transaction history) to identify customers likely to leave the bank, enabling proactive retention strategies.

**Business Problem:** Predicting which customers will churn (leave the bank) based on their profile and behavior.

**Solution:** An optimized ANN that achieves 78.55% accuracy with balanced precision and recall.

---

## 📊 Dataset

**Source:** Churn_Modelling.csv

**Features (13 input variables):**
- Credit Score
- Geography (France, Germany, Spain)
- Gender (Male, Female)
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary

**Target Variable:** 
- Exited (0 = Stayed, 1 = Left)

**Dataset Size:** 10,000 customers

---

## 🛠 Technologies Used

- **Python 3.x**
- **TensorFlow/Keras** - Deep learning framework
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation
- **Scikit-learn** - Data preprocessing and metrics
- **Matplotlib & Seaborn** - Data visualization
- **imbalanced-learn** - SMOTE for handling class imbalance

---

## 🔄 Project Workflow

### 1. **Data Preprocessing**
   - Load and explore dataset
   - Remove unnecessary columns (RowNumber, CustomerId, Surname)
   - Handle missing values (none found)
   - One-hot encoding for categorical variables (Geography, Gender)
   - Feature scaling using StandardScaler
   - Train-test split (80/20)

### 2. **Exploratory Data Analysis (EDA)**
   - Distribution analysis with histograms
   - Customer churn visualization
   - Credit card ownership analysis
   - Active member analysis
   - Product count analysis

### 3. **Handling Class Imbalance**
   - Applied SMOTE (Synthetic Minority Over-sampling Technique)
   - Balanced minority class (churned customers)
   - k_neighbors=5 for synthetic sample generation

### 4. **Model Building**
   - Sequential ANN architecture
   - Dense layers with ReLU activation
   - Dropout layers for regularization
   - Sigmoid activation for binary classification
   - Adam optimizer
   - Binary cross-entropy loss function

### 5. **Model Training**
   - Batch size: 10
   - Max epochs: 200
   - Early stopping with patience=15
   - Restored best weights

### 6. **Model Evaluation**
   - Confusion matrix visualization
   - Accuracy, Precision, Recall, F1-Score
   - Training history visualization

---

## 🏗 Model Architecture

```python
classifier = Sequential()

# Input Layer + First Hidden Layer
classifier.add(Dense(units=16, activation='relu', input_dim=13))
classifier.add(Dropout(0.3))

# Second Hidden Layer
classifier.add(Dense(units=8, activation='relu'))
classifier.add(Dropout(0.3))

# Output Layer
classifier.add(Dense(units=1, activation='sigmoid'))

# Compilation
classifier.compile(optimizer='adam', 
                   metrics=['accuracy'], 
                   loss='binary_crossentropy')
```

**Architecture Details:**
- **Input Layer:** 13 features
- **Hidden Layer 1:** 16 neurons, ReLU activation, 30% dropout
- **Hidden Layer 2:** 8 neurons, ReLU activation, 30% dropout
- **Output Layer:** 1 neuron, Sigmoid activation (binary classification)
- **Total Parameters:** ~400 trainable parameters

---

## 📈 Performance Metrics

### Final Model Results:

| Metric | Score |
|--------|-------|
| **Accuracy** | 78.55% |
| **Precision** | 48.12% |
| **Recall** | 75.80% |
| **F1 Score** | 58.87% |

### Model Improvements:

| Metric | Before Optimization | After Optimization | Improvement |
|--------|--------------------|--------------------|-------------|
| Accuracy | 73.00% | 78.55% | +5.55% |
| Precision | 41.59% | 48.12% | +6.53% |
| Recall | 82.47% | 75.80% | -6.67% |
| F1 Score | 55.30% | 58.87% | +3.57% |

### Optimization Techniques Applied:
1. ✅ Increased model capacity (16 → 8 units)
2. ✅ Added dropout layers (30% rate)
3. ✅ Implemented early stopping
4. ✅ Applied SMOTE for class balancing
5. ✅ Feature scaling with StandardScaler

---

## 💻 Installation

### Prerequisites
```bash
Python 3.8+
pip or conda package manager
```

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/DeepLearning-ANN-Build-and-Optimize.git
cd DeepLearning-ANN-Build-and-Optimize

# Install required packages
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn imbalanced-learn

# Or using requirements.txt
pip install -r requirements.txt
```

---

## 🚀 Usage

### Running the Notebook
```bash
# Launch Jupyter Notebook
jupyter notebook ANN_project.ipynb

# Or use VS Code with Jupyter extension
code ANN_project.ipynb
```

### Making Predictions
```python
# Single customer prediction
new_customer = [[608, 41, 1, 83807.86, 1, 0, 1, 112542.58, 0, 0, 1, 1, 0]]
prediction = classifier.predict(sc.transform(np.array(new_customer)))

if prediction > 0.5:
    print("Customer will LEAVE the bank (Churn)")
else:
    print("Customer will STAY with the bank")
```

---

## 🎯 Results

### Key Findings:
1. **Model successfully predicts churn** with 78.55% accuracy
2. **High recall (75.80%)** - catches most customers who will churn
3. **Balanced performance** with 58.87% F1 score
4. **Dropout and early stopping** effectively prevent overfitting
5. **SMOTE** improves model performance on imbalanced data

### Business Impact:
- Identify at-risk customers for targeted retention campaigns
- Reduce customer acquisition costs by retaining existing customers
- Improve customer lifetime value
- Data-driven decision making for marketing strategies

---

## 🧠 Key Learnings

### Technical Skills:
- Building and training neural networks with TensorFlow/Keras
- Handling imbalanced datasets with SMOTE
- Implementing regularization techniques (Dropout, Early Stopping)
- Feature engineering and preprocessing
- Model evaluation and optimization

### Best Practices:
- Always scale features for neural networks
- Use early stopping to prevent overfitting
- Monitor multiple metrics (not just accuracy)
- Visualize training history to detect issues
- Balance precision and recall based on business needs

### Future Improvements:
- Experiment with different architectures (more layers, different units)
- Try alternative optimizers (RMSprop, SGD)
- Implement cross-validation
- Feature selection and engineering
- Hyperparameter tuning with Grid Search or Random Search
- Deploy model as REST API

---

## 📁 Project Structure

```
DeepLearning_ANN_Build_&_Optimize/
│
├── ANN_project.ipynb          # Main Jupyter notebook
├── Churn_Modelling.csv        # Dataset
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies
```

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

---

## 🙏 Acknowledgments

- Dataset: Bank Customer Churn Dataset
- TensorFlow/Keras Documentation
- Scikit-learn Documentation
- Deep Learning community

---

## 📧 Contact

For questions or feedback, please reach out via [email@example.com](mailto:email@example.com)

---

**⭐ If you found this project helpful, please consider giving it a star!**