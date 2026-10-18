
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. Load dataset
data = pd.read_csv("dataset/student_queries.csv")

X = data["Query"]
y = data["Category"]

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Create NLP model
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("classifier", MultinomialNB())
])

# 4. Train model
model.fit(X_train, y_train)

# 5. Make predictions
predictions = model.predict(X_test)

# 6. Display accuracy
accuracy = accuracy_score(y_test, predictions)
print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# 7. Display classification report
print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

# 8. Generate confusion matrix
labels = model.named_steps["classifier"].classes_

cm = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)

print("\nConfusion Matrix:")
print(pd.DataFrame(
    cm,
    index=labels,
    columns=labels
))

# 9. Save confusion matrix as an image
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

display.plot(cmap="Blues", values_format="d")
plt.title("Student Query Categorization - Confusion Matrix")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()

# 10. Save trained model
with open("student_query_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nConfusion matrix saved as confusion_matrix.png")
print("Model trained successfully!")

