import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(r"C:\Users\HUAWEI\OneDrive\Desktop\gym_members data.csv")

x = df[["visits_per_week", "months_active", "last_visit_days"]]
y = df["churn"]

model = RandomForestClassifier()
model.fit(x, y)

print("Model trained successfully!")

def predict_churn(visits, months, last_visit):
    data = pd.DataFrame([[visits, months, last_visit]],
                        columns=["visits_per_week", "months_active", "last_visit_days"])
    prediction = model.predict(data)

    if prediction[0] == 1:
        return "User will leave gym (CHURN)"
    else:
        return "User will stay"

# test
print("Test 1:", predict_churn(5, 12, 1))
print("Test 2:", predict_churn(1, 2, 30))
print("Test 3:", predict_churn(4, 8, 2))
