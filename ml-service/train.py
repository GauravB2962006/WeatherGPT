"""Bootstrap model for the WeatherGPT MVP.
Replace the generated training data with a real historical weather dataset before claiming real-world model performance.
"""
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

rng = np.random.default_rng(42)
n = 4000
rainfall = rng.gamma(shape=2.0, scale=18.0, size=n)
rain_prob = np.clip(rng.normal(55 + rainfall * 0.45, 20, n), 0, 100)
wind = np.clip(rng.normal(18 + rainfall * 0.15, 10, n), 0, 90)
humidity = np.clip(rng.normal(68 + rainfall * 0.25, 12, n), 20, 100)
temp = np.clip(rng.normal(27, 5, n), 10, 45)
pressure = np.clip(rng.normal(1010 - rainfall * 0.12, 5, n), 980, 1030)

score = rainfall * 0.045 + rain_prob * 0.02 + wind * 0.012 + humidity * 0.004 + (1010 - pressure) * 0.02
risk = np.where(score >= 4.0, 2, np.where(score >= 2.2, 1, 0))
X = np.column_stack([temp, humidity, rainfall, rain_prob, wind, pressure])
features = ['temperature', 'humidity', 'rainfall', 'rain_probability', 'wind_speed', 'pressure']

X_train, X_test, y_train, y_test = train_test_split(X, risk, test_size=0.2, random_state=42, stratify=risk)
model = RandomForestClassifier(n_estimators=250, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test), target_names=['Low','Moderate','High']))

out = Path(__file__).parent / 'risk_model.joblib'
joblib.dump({'model': model, 'features': features, 'labels': {0:'Low',1:'Moderate',2:'High'}}, out)
print(f'Saved {out}')
