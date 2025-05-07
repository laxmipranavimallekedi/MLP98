# Health Insights – Predict Sleep Quality & Heart Rate

A machine learning–powered web application that predicts a user’s **sleep quality** and **resting heart rate** based on their lifestyle habits such as sleep duration, stress, activity level, and steps. The app provides instant predictions along with personalized health tips through an interactive web interface.

---

## 🧠 Features

* Predicts **Sleep Quality (scale 1–10)**
* Predicts **Heart Rate (in bpm)**
* Offers **personalized health tips** based on predictions
* Built with a **trained machine learning model**
* Fully responsive **web interface**
* **Popup modal** shows results with dynamic feedback
* Deployed on **Railway** for public access

---

## 💡 How It Works

1. User enters 5 daily routine inputs:

   * Age
   * Sleep duration (hours)
   * Physical activity (minutes)
   * Stress level (1–10)
   * Steps taken per day
2. Data is sent to the backend via JavaScript `fetch()`
3. The Flask server loads a trained Random Forest model and returns:

   * Sleep quality prediction
   * Heart rate prediction
   * Personalized advice
4. The frontend displays results in a **popup modal** without reloading the page

---

## 🛠️ Tech Stack

| Layer            | Tools Used                        |
| ---------------- | --------------------------------- |
| Machine Learning | `scikit-learn`, `pandas`, `numpy` |
| Backend          | `Flask`, `pickle`                 |
| Frontend         | `HTML`, `CSS`, `JavaScript`       |
| Deployment       | `Railway`                         |

---

## 📁 Folder Structure

```
health-insights-app/
├── app/
│   ├── app.py             # Flask server
│   └── utils.py           # Prediction + health tips
├── models/
│   └── sleep_model.pkl    # Trained ML model
├── templates/
│   └── index.html         # HTML frontend
├── static/
│   └── styles.css         # Styling
├── requirements.txt       # Python dependencies
├── Procfile               # Railway deployment config
```

---

## 🚀 Deployment

Live App: [Click here to view the deployed app](https://web-production-75448.up.railway.app)

Deployed using [Railway](https://railway.app):

1. Clone this repo
2. Make sure your model is saved in `models/sleep_model.pkl`
3. Use this `Procfile`:

   ```
   web: python app/app.py
   ```
4. Ensure `requirements.txt` includes:

   ```
   Flask
   scikit-learn
   pandas
   numpy
   ```

---

## 👥 Team Members and Roles

* **Amisha Meka** – Responsible for training the machine learning model and preparing the dataset.
* **Laxmi Pranavi Mallekedi** – Developed the Flask backend, including the API routes and model integration.
* **Sri Lahari Katla** – Handled frontend form creation and JavaScript integration with the backend.
* **Vaishnavi Tapetla** – Focused on UI/UX design with CSS and deployed the project using Railway.

---

## ✅ What We Learned

* Real-world application of **multi-output regression**
* Model training and evaluation
* Flask backend routing and APIs
* Frontend and backend integration
* Deployment of machine learning apps to the cloud

---

## 📸 Demo

To run locally:

```bash
pip install -r requirements.txt
python app/app.py
```


## 📄 License

This project is for educational use and demonstration purposes only.
