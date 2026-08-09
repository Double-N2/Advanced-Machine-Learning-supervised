import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

data = pd.read_csv('most_streamed_spotify_2025.csv')
data.drop_duplicates(inplace=True)
data.reset_index(inplace=True)
data['Top_10'] = data['wrapped_global_top10_rank'].notna().astype(int)
print(data.isnull().sum())
print(data.head())
print(data.columns)
data.info()


# Transforming the string into a computer understandable format

numeric = [
    'rank',
    'billed_artist_count',
    'spotify_streams_total',
    'daily_streams',
    'daily_streams_rank',
    'daily_stream_share_pct'
]

categorical = [
    'track',
    'artist',
    'is_collaboration'
]

preprocess_string = ColumnTransformer(
    transformers=[
        (
            'Numeric',
            StandardScaler(),
            numeric
        ),
        (
            'Categorical',
            OneHotEncoder(handle_unknown='ignore'),
            categorical
        )

    ]
)
x = data.drop(['wrapped_global_top10_rank','Top_10'],axis=1)
y = data['Top_10']

# Training the model with Random Forest
model = Pipeline(steps=[
    ('preprocessing', preprocess_string),
    ('model',RandomForestClassifier(n_estimators=100,random_state=42)),

])
# Training the model with Decision Tree Classifier
models = Pipeline(steps=[
    ('preprocessing', preprocess_string),
    ('models', DecisionTreeClassifier(random_state=42)),
])

# Training with Logistic Regressor
model_prediction = Pipeline(steps=[
    ('preprocessing', preprocess_string),
    ('models', LogisticRegression(random_state=42)),
])

x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y # Use stratify = y when dealing with Classes
)

model.fit(x_train,y_train)
predictions = model.predict(x_test)
accuracy = accuracy_score(y_test, predictions)

models.fit(x_train,y_train)
prediction = models.predict(x_test)
accuracies = accuracy_score(y_test, prediction)

model_prediction.fit(x_train,y_train)
prediction_prediction = model_prediction.predict(x_test)
accurate = accuracy_score(y_test, prediction_prediction)

print("Random Forest Regressor: ",predictions)
print("Accuracy: ",accuracy)
print("\n")
print("Decision Tree Regressor: ",prediction)
print("Accuracy: ",accuracies)
print("\n")
print("Logistic Regression Regressor: ",prediction_prediction)
print("Accuracy: ",accurate)
print("\n")