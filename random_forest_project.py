import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# تحميل بيانات Titanic
df = sns.load_dataset('titanic')
df = df[['survived', 'pclass', 'sex', 'age', 'fare', 'embarked']]
df.dropna(inplace=True)

# تحويل البيانات النصية إلى أرقام
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# تحديد البيانات المستقلة والتابعة
X = df.drop('survived', axis=1)
y = df['survived']

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تدريب نموذج Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# التنبؤ
y_pred = model.predict(X_test)

# تقييم النموذج
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# مصفوفة الالتباس
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# أهمية الخصائص
importances = model.feature_importances_
feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feat_df = feat_df.sort_values(by='Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=feat_df)
plt.title("Feature Importance")
plt.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# تدريب النموذج
model = RandomForestClassifier(n_estimators=10)
model.fit(X_train, y_train)

# رسم أول شجرة
plt.figure(figsize=(20, 10))
plot_tree(model.estimators_[0], filled=True, feature_names=X.columns, class_names=['Not Survived', 'Survived'])
plt.show()

import seaborn as sns
df = sns.load_dataset('titanic')
df = df[['pclass', 'sex', 'age', 'fare', 'embarked', 'survived']]
print(df.head())  # يطبع أول 5 صفوف من البيانات

# لحفظ الصورة:
import pandas as pd
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 2))
ax.axis('off')
tbl = pd.plotting.table(ax, df.head(), loc='center')
plt.savefig("dataset_sample.png", bbox_inches='tight')
plt.show()
