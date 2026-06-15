import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("spam.csv")

print("Dataset Loaded Successfully!\n")
print(df.head())

# ==========================
# Convert Labels
# ham = 0
# spam = 1
# ==========================
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# ==========================
# Features and Target
# ==========================
X = df["text"]
y = df["label"]

# ==========================
# Convert Text to Numbers
# ==========================
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(X)

# ==========================
# Split Dataset
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# Train Model
# ==========================
model = MultinomialNB()

model.fit(X_train, y_train)

# ==========================
# Predict
# ==========================
y_pred = model.predict(X_test)

# ==========================
# Accuracy
# ==========================
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================
# User Testing
# ==========================
while True:

    print("\n----------------------")
    email = input("Enter Email Message: ")

    email_vector = vectorizer.transform([email])

    prediction = model.predict(email_vector)

    if prediction[0] == 1:
        print("Prediction: SPAM EMAIL")
    else:
        print("Prediction: HAM EMAIL")

    choice = input("\nCheck another email? (yes/no): ")

    if choice.lower() != "yes":
        print("Program Ended.")
        break