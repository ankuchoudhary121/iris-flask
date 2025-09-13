from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=50, random_state=42))
clf.fit(X_train, y_train)

print("Test accuracy:", clf.score(X_test, y_test))
joblib.dump(clf, "model.joblib")
print("Saved model to model.joblib")
