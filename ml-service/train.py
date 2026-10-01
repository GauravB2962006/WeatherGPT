"""
WeatherGPT - V4 Next-Day Risk Prediction

V4 improvements:
- Current weather features
- Previous-day weather features
- Seasonal features
- Chronological train/test split
- Class balancing
- Probability output
- Feature importance

Risk categories are project-defined engineering labels.
They are NOT official IMD warning categories.
"""

import json
import urllib.error
import urllib.parse
import urllib.request

from pathlib import Path
from datetime import datetime, timedelta

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

LATITUDE = 20.0059
LONGITUDE = 73.7897

LOCATION_NAME = "Nashik, Maharashtra, India"

START_DATE = "2020-01-01"
END_DATE = "2026-08-31"

TIMEZONE = "Asia/Kolkata"

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "risk_model.joblib"

DATA_PATH = (
    BASE_DIR /
    "historical_weather_v4_nashik.csv"
)

METADATA_PATH = (
    BASE_DIR /
    "model_metadata_v4.json"
)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [

    # Current weather
    "temperature",
    "humidity",
    "rainfall",
    "wind_speed",
    "pressure",

    # Previous-day weather
    "previous_rainfall",
    "previous_wind_speed",
    "previous_humidity",

    # Seasonal information
    "month",
    "day_of_year",
]


# ============================================================
# DOWNLOAD JSON
# ============================================================

def download_json(url):

    print("\nDownloading:")
    print(url)

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "WeatherGPT/1.0"
        }
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=180
        ) as response:

            raw = response.read().decode(
                "utf-8"
            )

        data = json.loads(raw)

        if data.get("error"):

            raise RuntimeError(
                data.get(
                    "reason",
                    "Open-Meteo returned an error."
                )
            )

        return data

    except urllib.error.HTTPError as error:

        print("\nOpen-Meteo HTTP ERROR")
        print("Status:", error.code)

        try:

            body = error.read().decode(
                "utf-8"
            )

            print("Server response:")
            print(body)

        except Exception:
            pass

        raise

    except Exception as error:

        print("\nDownload error:")
        print(error)

        raise


# ============================================================
# DATE CHUNKS
# ============================================================

def generate_chunks():

    start = datetime.strptime(
        START_DATE,
        "%Y-%m-%d"
    ).date()

    end = datetime.strptime(
        END_DATE,
        "%Y-%m-%d"
    ).date()

    chunks = []

    current = start

    while current <= end:

        chunk_end = min(
            current + timedelta(days=364),
            end
        )

        chunks.append(
            (
                current.strftime(
                    "%Y-%m-%d"
                ),
                chunk_end.strftime(
                    "%Y-%m-%d"
                )
            )
        )

        current = (
            chunk_end +
            timedelta(days=1)
        )

    return chunks


# ============================================================
# DOWNLOAD ONE HISTORICAL CHUNK
# ============================================================

def download_historical_chunk(
    start_date,
    end_date
):

    daily_variables = (
        "temperature_2m_max,"
        "relative_humidity_2m_mean,"
        "precipitation_sum,"
        "wind_speed_10m_max,"
        "surface_pressure_mean,"
        "weather_code"
    )

    params = {

        "latitude":
            LATITUDE,

        "longitude":
            LONGITUDE,

        "start_date":
            start_date,

        "end_date":
            end_date,

        "daily":
            daily_variables,

        "timezone":
            TIMEZONE,

        "temperature_unit":
            "celsius",

        "wind_speed_unit":
            "kmh",

        "precipitation_unit":
            "mm",
    }

    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        +
        urllib.parse.urlencode(params)
    )

    data = download_json(url)

    if "daily" not in data:

        raise RuntimeError(
            "No daily data returned."
        )

    daily = data["daily"]

    required_columns = [

        "time",

        "temperature_2m_max",

        "relative_humidity_2m_mean",

        "precipitation_sum",

        "wind_speed_10m_max",

        "surface_pressure_mean",

        "weather_code",
    ]

    for column in required_columns:

        if column not in daily:

            raise RuntimeError(
                f"Missing API field: {column}"
            )

    df = pd.DataFrame(daily)

    df = df.rename(
        columns={

            "time":
                "date",

            "temperature_2m_max":
                "temperature",

            "relative_humidity_2m_mean":
                "humidity",

            "precipitation_sum":
                "rainfall",

            "wind_speed_10m_max":
                "wind_speed",

            "surface_pressure_mean":
                "pressure",

            "weather_code":
                "weather_code",
        }
    )

    return df


# ============================================================
# DOWNLOAD ALL DATA
# ============================================================

def download_historical_data():

    print("\n")
    print("=" * 70)
    print(
        "WEATHERGPT V4"
    )
    print(
        "DOWNLOADING REAL HISTORICAL WEATHER DATA"
    )
    print("=" * 70)

    chunks = generate_chunks()

    all_data = []

    for index, (
        start_date,
        end_date
    ) in enumerate(
        chunks,
        start=1
    ):

        print(
            f"\nChunk {index}/{len(chunks)}"
        )

        print(
            f"Period: "
            f"{start_date} → {end_date}"
        )

        df = (
            download_historical_chunk(
                start_date,
                end_date
            )
        )

        print(
            f"Rows received: "
            f"{len(df)}"
        )

        all_data.append(df)

    if not all_data:

        raise RuntimeError(
            "No historical weather data downloaded."
        )

    df = pd.concat(
        all_data,
        ignore_index=True
    )

    df = df.drop_duplicates(
        subset=["date"]
    )

    df = df.sort_values(
        "date"
    )

    df = df.reset_index(
        drop=True
    )

    return df


# ============================================================
# CREATE RISK LABEL
# ============================================================

def create_risk_label(row):

    rainfall = float(
        row["rainfall"]
    )

    wind = float(
        row["wind_speed"]
    )

    weather_code = int(
        row["weather_code"]
    )

    high_codes = {
        95,
        96,
        99,
    }

    medium_codes = {
        51,
        53,
        55,
        56,
        57,
        61,
        63,
        65,
        66,
        67,
        80,
        81,
        82,
    }

    # HIGH
    if (
        rainfall >= 20
        or wind >= 45
        or weather_code in high_codes
    ):

        return "High"

    # MEDIUM
    if (
        rainfall >= 5
        or wind >= 30
        or weather_code in medium_codes
    ):

        return "Medium"

    return "Low"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "WeatherGPT - V4 NEXT-DAY RISK MODEL"
    )
    print("=" * 70)

    print(
        f"\nLocation : "
        f"{LOCATION_NAME}"
    )

    print(
        f"Latitude : "
        f"{LATITUDE}"
    )

    print(
        f"Longitude: "
        f"{LONGITUDE}"
    )

    print(
        f"Period   : "
        f"{START_DATE} → {END_DATE}"
    )


    # ========================================================
    # STEP 1
    # DOWNLOAD
    # ========================================================

    df = download_historical_data()

    print("\n")
    print("=" * 70)
    print(
        "HISTORICAL DATA DOWNLOADED"
    )
    print("=" * 70)

    print(
        f"\nTotal rows: {len(df)}"
    )


    # ========================================================
    # STEP 2
    # CLEAN
    # ========================================================

    print("\n")
    print("=" * 70)
    print("CLEANING DATA")
    print("=" * 70)

    df["date"] = pd.to_datetime(
        df["date"]
    )

    numeric_columns = [

        "temperature",
        "humidity",
        "rainfall",
        "wind_speed",
        "pressure",
        "weather_code",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    before = len(df)

    df = df.dropna(
        subset=numeric_columns
    )

    print(
        f"Rows removed: "
        f"{before - len(df)}"
    )

    print(
        f"Clean rows: "
        f"{len(df)}"
    )


    # ========================================================
    # STEP 3
    # CREATE CURRENT RISK
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "CREATING RISK LABELS"
    )
    print("=" * 70)

    df["current_risk"] = (
        df.apply(
            create_risk_label,
            axis=1
        )
    )


    # ========================================================
    # STEP 4
    # PREVIOUS-DAY FEATURES
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "CREATING PREVIOUS-DAY FEATURES"
    )
    print("=" * 70)

    df["previous_rainfall"] = (
        df["rainfall"].shift(1)
    )

    df["previous_wind_speed"] = (
        df["wind_speed"].shift(1)
    )

    df["previous_humidity"] = (
        df["humidity"].shift(1)
    )


    # ========================================================
    # STEP 5
    # SEASONAL FEATURES
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "CREATING SEASONAL FEATURES"
    )
    print("=" * 70)

    df["month"] = (
        df["date"].dt.month
    )

    df["day_of_year"] = (
        df["date"].dt.dayofyear
    )


    # ========================================================
    # STEP 6
    # NEXT-DAY TARGET
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "CREATING NEXT-DAY TARGET"
    )
    print("=" * 70)

    df["target_risk"] = (
        df["current_risk"].shift(-1)
    )


    # ========================================================
    # STEP 7
    # REMOVE INVALID ROWS
    # ========================================================

    df = df.dropna(
        subset=FEATURES + [
            "target_risk"
        ]
    )

    df = df.reset_index(
        drop=True
    )

    print(
        f"\nRows available for ML: "
        f"{len(df)}"
    )


    # ========================================================
    # STEP 8
    # DISTRIBUTION
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "NEXT-DAY RISK DISTRIBUTION"
    )
    print("=" * 70)

    print(
        df["target_risk"].value_counts()
    )


    # ========================================================
    # STEP 9
    # SAVE DATASET
    # ========================================================

    df.to_csv(
        DATA_PATH,
        index=False
    )

    print(
        f"\nDataset saved:"
    )

    print(
        DATA_PATH
    )


    # ========================================================
    # STEP 10
    # PREPARE DATA
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "PREPARING TRAIN / TEST DATA"
    )
    print("=" * 70)

    X = df[FEATURES]

    y = df["target_risk"]

    split_index = int(
        len(df) * 0.80
    )

    X_train = X.iloc[
        :split_index
    ]

    X_test = X.iloc[
        split_index:
    ]

    y_train = y.iloc[
        :split_index
    ]

    y_test = y.iloc[
        split_index:
    ]

    print(
        f"\nTraining rows: "
        f"{len(X_train)}"
    )

    print(
        f"Testing rows : "
        f"{len(X_test)}"
    )

    print(
        "\nTraining period:"
    )

    print(
        f"{df.iloc[0]['date'].date()}"
        f" → "
        f"{df.iloc[split_index - 1]['date'].date()}"
    )

    print(
        "\nTesting period:"
    )

    print(
        f"{df.iloc[split_index]['date'].date()}"
        f" → "
        f"{df.iloc[-1]['date'].date()}"
    )


    # ========================================================
    # STEP 11
    # RANDOM FOREST V4
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "TRAINING RANDOM FOREST V4"
    )
    print("=" * 70)

    class_weights = {

        "Low":
            0.8,

        "Medium":
            1.0,

        "High":
            2.5,
    }

    model = RandomForestClassifier(

        n_estimators=500,

        max_depth=16,

        min_samples_split=4,

        min_samples_leaf=2,

        max_features="sqrt",

        class_weight=class_weights,

        random_state=42,

        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "\nRandom Forest V4 training completed."
    )


    # ========================================================
    # STEP 12
    # PREDICTION
    # ========================================================

    predictions = (
        model.predict(
            X_test
        )
    )

    probabilities = (
        model.predict_proba(
            X_test
        )
    )


    # ========================================================
    # STEP 13
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    print("\n")
    print("=" * 70)
    print(
        "MODEL EVALUATION"
    )
    print("=" * 70)

    print(
        f"\nAccuracy : "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision: "
        f"{precision:.4f}"
    )

    print(
        f"Recall   : "
        f"{recall:.4f}"
    )

    print(
        f"F1 Score : "
        f"{f1:.4f}"
    )


    # ========================================================
    # STEP 14
    # CLASSIFICATION REPORT
    # ========================================================

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            labels=[
                "Low",
                "Medium",
                "High",
            ],
            zero_division=0
        )
    )


    # ========================================================
    # STEP 15
    # CONFUSION MATRIX
    # ========================================================

    labels = [
        "Low",
        "Medium",
        "High",
    ]

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        pd.DataFrame(
            matrix,
            index=[
                "Actual Low",
                "Actual Medium",
                "Actual High",
            ],
            columns=[
                "Predicted Low",
                "Predicted Medium",
                "Predicted High",
            ],
        )
    )


    # ========================================================
    # STEP 16
    # HIGH-RISK METRICS
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "HIGH-RISK DETECTION"
    )
    print("=" * 70)

    high_actual = (
        y_test == "High"
    )

    high_predicted = (
        predictions == "High"
    )

    high_true_positive = (
        high_actual &
        high_predicted
    ).sum()

    high_total = (
        high_actual
    ).sum()

    if high_total > 0:

        high_recall = (
            high_true_positive /
            high_total
        )

    else:

        high_recall = 0.0

    print(
        f"Actual High cases    : "
        f"{high_total}"
    )

    print(
        f"Correct High cases   : "
        f"{high_true_positive}"
    )

    print(
        f"High-risk recall     : "
        f"{high_recall:.4f}"
    )


    # ========================================================
    # STEP 17
    # FEATURE IMPORTANCE
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "FEATURE IMPORTANCE"
    )
    print("=" * 70)

    importance_df = pd.DataFrame(
        {

            "feature":
                FEATURES,

            "importance":
                model.feature_importances_,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
    )

    for _, row in (
        importance_df.iterrows()
    ):

        print(
            f"{row['feature']:25s}"
            f"{row['importance']:.4f}"
        )


    # ========================================================
    # STEP 18
    # SAVE MODEL
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "SAVING V4 MODEL"
    )
    print("=" * 70)

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        "\nModel saved:"
    )

    print(
        MODEL_PATH
    )


    # ========================================================
    # STEP 19
    # SAVE METADATA
    # ========================================================

    metadata = {

        "project":
            "WeatherGPT",

        "model_version":
            "V4",

        "model_type":
            "RandomForestClassifier",

        "prediction_task":
            "Predict next-day weather risk",

        "location":
            LOCATION_NAME,

        "latitude":
            LATITUDE,

        "longitude":
            LONGITUDE,

        "timezone":
            TIMEZONE,

        "training_start":
            START_DATE,

        "training_end":
            END_DATE,

        "data_source":
            "Open-Meteo Historical Weather API",

        "features":
            FEATURES,

        "target":
            "next_day_risk",

        "risk_definition": {

            "High":
                (
                    "Rainfall >= 20 mm OR "
                    "wind >= 45 km/h OR "
                    "thunderstorm"
                ),

            "Medium":
                (
                    "Rainfall >= 5 mm OR "
                    "wind >= 30 km/h OR "
                    "moderate rain/showers"
                ),

            "Low":
                "All remaining conditions",
        },

        "important_note":
            (
                "Risk labels are project-defined "
                "engineering categories and are not "
                "official IMD warning categories."
            ),

        "method":
            (
                "Current weather, previous-day weather "
                "and seasonal features are used to "
                "predict the following day's risk."
            ),

        "train_test_split":
            "Chronological 80/20",

        "parameters": {

            "n_estimators":
                500,

            "max_depth":
                16,

            "min_samples_split":
                4,

            "min_samples_leaf":
                2,

            "max_features":
                "sqrt",

            "class_weight":
                class_weights,

            "random_state":
                42,
        },

        "metrics": {

            "accuracy":
                float(accuracy),

            "macro_precision":
                float(precision),

            "macro_recall":
                float(recall),

            "macro_f1":
                float(f1),

            "high_recall":
                float(high_recall),
        },

        "feature_importance": {

            row["feature"]:
                float(row["importance"])

            for _, row
            in importance_df.iterrows()
        },
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print(
        "\nMetadata saved:"
    )

    print(
        METADATA_PATH
    )


    # ========================================================
    # DONE
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "V4 TRAINING COMPLETED"
    )
    print("=" * 70)

    print(
        "\nGenerated files:"
    )

    print(
        "✓ risk_model.joblib"
    )

    print(
        "✓ historical_weather_v4_nashik.csv"
    )

    print(
        "✓ model_metadata_v4.json"
    )

    print(
        "\nThe model predicts NEXT-DAY risk."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()