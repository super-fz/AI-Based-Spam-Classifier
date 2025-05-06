from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

texts = ["Win a car now!", "Hey, how are you?", "Congratulations! You've won", "Let’s meet for lunch"]
labels = [1, 0, 1, 0]  # 1 = spam, 0 = not spam

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

test_text = [input("Enter a message to classify: ")]
print("SPAM" if model.predict(vectorizer.transform(test_text))[0] else "NOT SPAM")
