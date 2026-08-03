# 🌤️ ClimaPulse AI — Climate Intelligence & ML Forecasting Dashboard

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

ClimaPulse AI is an end-to-end, interactive weather intelligence platform designed to deliver real-time atmospheric telemetry, short-term (16-day) physical weather forecasts, and multi-year (5-year) machine learning climate projections using Random Forest Regressors.

---

## 🔥 Key Features

- **📍 Real-Time Telemetry & 16-Day Forecast:** Live data fetching via Open-Meteo API with automated local timezone recognition and daily weather breakdowns.
- **🏃 Outdoor Lifestyle & Activity Index:** Custom scoring algorithms evaluating real-time condition suitability for Sports, Laundry Drying, Commuting, and Travel.
- **🔮 5-Year ML Climate Projections:** Powered by `RandomForestRegressor` with cyclic feature engineering ($sin/cos$ transformation on day-of-year) to project long-term monthly climate baselines.
- **🚨 Multi-Channel Automated Alerts:** Real-time dynamic monitoring engine triggering automated alerts via **WhatsApp** (Twilio API) and **Email** (SMTP).
- **📥 Scientific Data Export:** Filter and export custom historical datasets and ML predictions into **CSV** and formatted **PDF summary reports**.

---

## 🛠️ Tech Stack

- **Frontend / UI:** Streamlit (Custom Dark Charcoal Neumorphic CSS Theme)
- **Data Analytics & Viz:** Pandas, NumPy, Plotly Express
- **Machine Learning:** Scikit-Learn (`RandomForestRegressor`)
- **APIs & Integration:** Open-Meteo API, Twilio WhatsApp API, Python SMTP (`smtplib`)
- **Document Generation:** FPDF

---

├── .gitignore            # Git Exclusions File
└── README.md             # Project Documentation
