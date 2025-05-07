import pandas as pd
import numpy as np
import pickle
import os

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/sleep_model.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

FEATURE_NAMES = ['Age', 'Sleep_Duration', 'Activity', 'Stress', 'Steps']

def predict_sleep_health(age, sleep_duration, activity, stress, steps):
    input_df = pd.DataFrame([[age, sleep_duration, activity, stress, steps]], columns=FEATURE_NAMES)
    prediction = model.predict(input_df)[0]
    return {
        'sleep_quality': round(prediction[0], 2),
        'heart_rate': round(prediction[1], 2)
    }

def generate_tip(sleep_quality, heart_rate):
    # Analyze sleep quality
    if sleep_quality < 5:
        sleep_msg = "😴 Your sleep quality is low. Try to increase your sleep duration and reduce daily stress."
    elif 5 <= sleep_quality < 8:
        sleep_msg = "😐 Your sleep quality is moderate. Small improvements in your routine can make a big difference."
    else:
        sleep_msg = "😁 Excellent sleep quality! You're sleeping well. keep it up."

    # Analyze heart rate
    if heart_rate < 60:
        heart_msg = "💓 Your heart rate is lower than average. This could mean high fitness, but monitor for fatigue or dizziness."
    elif 60 <= heart_rate <= 85:
        heart_msg = "🫀 Your heart rate is within a healthy range. Great job maintaining balance!"
    else:
        heart_msg = "⚠️ Your heart rate is above average. Try adding light physical activity, relaxation, or reducing caffeine."

    return f"{sleep_msg} {heart_msg}"


