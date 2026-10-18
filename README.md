# Student Query Categorization Using NLP

## 1. Project Overview
Student Query Categorization is a Natural Language Processing (NLP) project that automatically classifies student questions into different categories using Machine Learning.

It helps organize student queries so that they can be directed to the appropriate department.

## 2. Objectives
- Automatically categorize student queries.
- Apply NLP techniques to text data.
- Train a machine learning model to classify questions.
- Display the predicted category through a simple web application.

## 3. Categories
The project classifies student queries into five categories:

- Attendance
- Examination
- Fees
- Placements
- Technical Support

## 4. Technologies Used
- **Python** – Programming language
- **Pandas** – Dataset handling
- **Scikit-learn** – Machine learning and text processing
- **TF-IDF** – Converts text into numerical features
- **Multinomial Naive Bayes** – Classification algorithm
- **Streamlit** – Web application interface
- **Matplotlib** – Confusion matrix visualization

## 5. Project Structure

```text
Student_Query_Categorization/
│
├── dataset/
│   └── student_queries.csv
│
├── train_model.py
├── app.py
├── student_query_model.pkl
├── confusion_matrix.png
└── README.md
```

## 6. How It Works
1. Student queries are collected in a CSV dataset.
2. The dataset contains queries and their categories.
3. TF-IDF converts the text into numerical features.
4. The Multinomial Naive Bayes algorithm learns from the training data.
5. The model predicts the category of a new student query.
6. Streamlit displays the prediction on a web page.

## 7. Installation

Install the required libraries by running:

```bash
python -m pip install pandas scikit-learn streamlit matplotlib
```

## 8. How to Run the Project

### Step 1: Train the Model

Open the VS Code terminal in the project folder and run:

```bash
python train_model.py
```

This trains the model, displays evaluation results, saves the trained model as `student_query_model.pkl`, and creates `confusion_matrix.png`.

### Step 2: Start the Web Application

Run:

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal, usually:

`http://localhost:8501`

Enter a student question to see its predicted category.

## 9. Model Evaluation
The current model achieved **90.38% accuracy** on a test set of 52 queries.

The model was evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Results may vary with different datasets and test splits. The accuracy on this small test set does not guarantee the same performance on new real-world questions.

## 10. Conclusion
This project demonstrates how Natural Language Processing and Machine Learning can be used to categorize student queries. The Streamlit interface makes it easy for users to enter questions and view the predicted categories.
