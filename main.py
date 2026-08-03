import os
import requests
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import smtplib
import plotly.express as px
from email.mime.text import MIMEText
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from fpdf import FPDF

# Load environment variables
load_dotenv()

# Configure page settings
st.set_page_config(
    page_title="Weather Dashboard & ML Forecast",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR DARK CHARCOAL / NEUMORPHIC MOOD THEME ---
st.markdown("""
<style>
    /* Main App Background Gradient (Deep Dark Charcoal) */
    .stApp {
        background: linear-gradient(160deg, #121319 0%, #1a1c26 50%, #222533 100%) !important;
        color: #f1f2f6;
    }

    /* Sidebar Custom Styling */
    [data-testid="stSidebar"] {
        background: rgba(20, 22, 30, 0.85) !important;
        backdrop-filter: blur(16px);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Headers, Captions & Labels */
    h1, h2, h3, h4, h5, h6, label, p {
        color: #ffffff !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    .stCaption, [data-testid="stMarkdownContainer"] p {
        color: #a0a5b5 !important;
    }

    /* Tabs Styling */
    button[data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.04) !important;
        border-radius: 12px 12px 0px 0px !important;
        color: #8c93a8 !important;
        font-weight: 600 !important;
        border: none !important;
        margin-right: 6px !important;
    }

    button[aria-selected="true"] {
        background-color: rgba(40, 44, 62, 0.9) !important;
        color: #ff9f43 !important;
        border-bottom: 3px solid #ff9f43 !important;
    }

    /* Input Field Overrides */
    input {
        background-color: rgba(15, 16, 22, 0.8) !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
</style>
""", unsafe_allow_html=True)


# --- TEXT-TO-SPEECH FUNCTION ---
def speak(text_to_speak):
    """Uses PowerShell System.Speech to read out text aloud."""
    try:
        clean_text = text_to_speak.replace('"', '').replace("'", "")
        command = f'PowerShell -Command "Add-Type –AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{clean_text}\')"'
        os.system(command)
    except Exception as e:
        st.warning(f"Audio playback issue: {e}")


# --- LIFESTYLE & OUTDOOR ACTIVITY SCORING ENGINE ---
def calculate_lifestyle_scores(in_temp, in_humidity, in_wind_speed, in_rain_prob):
    """Calculates 0-100 scores for various outdoor activities based on weather metrics."""
    sports = 100
    if in_temp > 35:
        sports -= (in_temp - 35) * 4
    elif in_temp < 15:
        sports -= (15 - in_temp) * 3
    sports -= (in_rain_prob * 0.7)
    if in_humidity > 70:
        sports -= (in_humidity - 70) * 0.5
    sports_score = int(np.clip(sports, 0, 100))

    laundry = 100
    if in_rain_prob > 15:
        laundry -= (in_rain_prob * 0.8)
    laundry -= (in_humidity * 0.4)
    if in_temp >= 25:
        laundry += 10
    elif in_temp < 15:
        laundry -= (15 - in_temp) * 2
    if 10 <= in_wind_speed <= 35:
        laundry += 10
    laundry_score = int(np.clip(laundry, 0, 100))

    commute = 100
    commute -= (in_rain_prob * 0.6)
    if in_wind_speed > 30:
        commute -= (in_wind_speed - 30) * 1.5
    if in_temp > 42 or in_temp < 5:
        commute -= 15
    commute_score = int(np.clip(commute, 0, 100))

    travel = 100
    if in_temp < 18:
        travel -= (18 - in_temp) * 2.5
    elif in_temp > 28:
        travel -= (in_temp - 28) * 3
    travel -= (in_rain_prob * 0.7)
    if in_wind_speed > 25:
        travel -= (in_wind_speed - 25) * 1.2
    travel_score = int(np.clip(travel, 0, 100))

    return {
        "Sports": sports_score,
        "Laundry": laundry_score,
        "Commuting": commute_score,
        "Travel": travel_score
    }


def get_score_status(val_score):
    """Returns color hex and status label for a given score."""
    if val_score >= 80:
        return "#00cec9", "Excellent"
    elif val_score >= 60:
        return "#ff9f43", "Good"
    elif val_score >= 40:
        return "#feca57", "Moderate"
    else:
        return "#ff6b6b", "Poor"


# --- EXPORT HELPER FUNCTIONS (CSV & PDF) ---
@st.cache_data
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Converts a DataFrame into CSV format encoded for download."""
    return df.to_csv(index=False).encode('utf-8')


def generate_pdf_report(df: pd.DataFrame, pdf_city_name: str, dataset_type: str) -> bytes:
    """Generates a PDF summary report from a DataFrame using FPDF."""
    pdf = FPDF()
    pdf.add_page()

    # Title Header
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, f"Weather Report: {pdf_city_name}", new_x="LMARGIN", new_y="NEXT", align="C")

    # Subheader / Metadata
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 6,
             f"Dataset Type: {dataset_type} | Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)

    # Summary Metrics Section
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Dataset Overview:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"- Total Records Exported: {len(df)}", new_x="LMARGIN", new_y="NEXT")

    temp_col = None
    for tbl_col in ["max_temp", "Avg Max Temp", "Predicted Avg Max", "temperature_2m_max"]:
        if tbl_col in df.columns:
            temp_col = tbl_col
            break

    if temp_col and not df.empty:
        avg_temp = df[temp_col].mean()
        max_temp = df[temp_col].max()
        min_temp = df[temp_col].min()
        pdf.cell(0, 6,
                 f"- Max Temp Range ({temp_col}): Min {min_temp:.1f}°C | Avg {avg_temp:.1f}°C | Peak {max_temp:.1f}°C",
                 new_x="LMARGIN", new_y="NEXT")

    pdf.ln(5)

    # Table Header
    pdf.set_font("Helvetica", "B", 9)
    cols = list(df.columns)[:6]
    col_width = 190 / len(cols)

    for pdf_col in cols:
        pdf.cell(col_width, 8, str(pdf_col)[:15], border=1, align="C")
    pdf.ln()

    # Table Rows
    pdf.set_font("Helvetica", "", 8)
    for _, row in df.head(25).iterrows():
        for pdf_col in cols:
            val = str(row[pdf_col])
            pdf.cell(col_width, 6, val[:15], border=1, align="C")
        pdf.ln()

    if len(df) > 25:
        pdf.ln(3)
        pdf.set_font("Helvetica", "I", 8)
        pdf.cell(0, 6, f"* Note: Previewing first 25 rows of {len(df)} total records. Download CSV for full dataset.",
                 new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())


# --- GEOCODING & REAL-TIME WEATHER FETCHING (OPEN-METEO) ---
def get_city_coordinates(target_city):
    """Fetches latitude, longitude, and full country name for any city."""
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={target_city}&count=1&language=en&format=json"
    try:
        res = requests.get(geo_url)
        if res.status_code == 200:
            results = res.json().get("results")
            if results:
                return results[0]["latitude"], results[0]["longitude"], results[0]["name"], results[0].get("country", ""), None
            return None, None, None, None, "City not found. Please check spelling."
        return None, None, None, None, f"Geocoding service error ({res.status_code})."
    except Exception as e:
        return None, None, None, None, f"Network error: {e}"


def get_current_and_16day_forecast(target_lat, target_lon):
    """Fetches real-time current weather and guaranteed 16-day daily forecast via Open-Meteo."""
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={target_lat}&longitude={target_lon}&"
        f"current=temperature_2m,relative_humidity_2m,apparent_temperature,wind_speed_10m,weather_code&"
        f"daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max&"
        f"timezone=auto"
    )
    try:
        res = requests.get(url)
        if res.status_code == 200:
            return res.json(), None
        return None, f"Failed to fetch forecast. Status code: {res.status_code}"
    except Exception as e:
        return None, f"Could not connect to service: {e}"


# --- HISTORICAL (5 YEARS) & ML FORECAST ENGINE (5 YEARS) ---
@st.cache_data(ttl=86400)
def fetch_history_and_forecast_5years(target_lat, target_lon):
    """Fetches last 5 years historical data and projects next 5 years weather using ML."""
    today = datetime.date.today()

    # 1. Fetch Past 5 Years Historical Data
    hist_start = today - datetime.timedelta(days=365 * 5)
    hist_end = today - datetime.timedelta(days=1)

    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={target_lat}&longitude={target_lon}&start_date={hist_start}&end_date={hist_end}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=auto"

    res = requests.get(url)
    if res.status_code != 200:
        return None, None, None, None, "Could not fetch historical climate data."

    hist_data = res.json()
    df_hist = pd.DataFrame({
        "date": pd.to_datetime(hist_data["daily"]["time"]),
        "max_temp": hist_data["daily"]["temperature_2m_max"],
        "min_temp": hist_data["daily"]["temperature_2m_min"],
        "precipitation": hist_data["daily"]["precipitation_sum"]
    }).dropna()

    # Feature Engineering
    df_hist["year"] = df_hist["date"].dt.year
    df_hist["month_num"] = df_hist["date"].dt.month
    df_hist["month_name"] = df_hist["date"].dt.strftime("%B")
    df_hist["day_of_year"] = df_hist["date"].dt.dayofyear
    df_hist["sin_day"] = np.sin(2 * np.pi * df_hist["day_of_year"] / 365.25)
    df_hist["cos_day"] = np.cos(2 * np.pi * df_hist["day_of_year"] / 365.25)

    # Historical Monthly Summary
    hist_monthly_df = df_hist.groupby(["year", "month_num", "month_name"]).agg(
        Avg_Max_Temp=("max_temp", "mean"),
        Avg_Min_Temp=("min_temp", "mean"),
        Total_Precipitation=("precipitation", "sum")
    ).reset_index().sort_values(by=["year", "month_num"])

    # 2. ML Training (Random Forest)
    x_train = df_hist[["year", "sin_day", "cos_day", "day_of_year"]]
    y_max = df_hist["max_temp"]
    y_min = df_hist["min_temp"]

    model_max = RandomForestRegressor(n_estimators=100, random_state=42)
    model_max.fit(x_train, y_max)

    model_min = RandomForestRegressor(n_estimators=100, random_state=42)
    model_min.fit(x_train, y_min)

    # 3. Generate Next 5-Year Future Dataset
    future_dates = pd.date_range(start=today, periods=365 * 5, freq="D")
    df_future = pd.DataFrame({"date": future_dates})
    df_future["year"] = df_future["date"].dt.year
    df_future["month_num"] = df_future["date"].dt.month
    df_future["month_name"] = df_future["date"].dt.strftime("%B")
    df_future["day_of_year"] = df_future["date"].dt.dayofyear
    df_future["sin_day"] = np.sin(2 * np.pi * df_future["day_of_year"] / 365.25)
    df_future["cos_day"] = np.cos(2 * np.pi * df_future["day_of_year"] / 365.25)

    x_future = df_future[["year", "sin_day", "cos_day", "day_of_year"]]
    df_future["predicted_max_temp"] = model_max.predict(x_future)
    df_future["predicted_min_temp"] = model_min.predict(x_future)

    # Future Monthly Summary
    future_monthly_df = df_future.groupby(["year", "month_num", "month_name"]).agg(
        Predicted_Avg_Max_Temp=("predicted_max_temp", "mean"),
        Predicted_Avg_Min_Temp=("predicted_min_temp", "mean")
    ).reset_index().sort_values(by=["year", "month_num"])

    return df_hist, hist_monthly_df, future_monthly_df, df_future, None


# --- ALERT DISPATCHERS ---
def send_whatsapp_alert(account_sid, auth_token, from_number, to_number, message):
    if not account_sid or not auth_token or not from_number or not to_number:
        return False, "Twilio WhatsApp credentials missing."

    formatted_from = from_number if from_number.startswith("whatsapp:") else f"whatsapp:{from_number}"
    formatted_to = to_number if to_number.startswith("whatsapp:") else f"whatsapp:{to_number}"

    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
    payload = {"From": formatted_from, "To": formatted_to, "Body": message}

    try:
        res = requests.post(url, data=payload, auth=(account_sid, auth_token))
        return (True, "WhatsApp alert sent successfully!") if res.status_code in [200, 201] else (
            False, f"Twilio API error: {res.json().get('message', res.text)}")
    except Exception as e:
        return False, f"Failed to send WhatsApp alert: {e}"


def send_email_alert(sender_email, sender_password, recipient_email, subject, body):
    if not sender_email or not sender_password or not recipient_email:
        return False, "Email credentials missing."
    try:
        clean_pass = sender_password.replace(" ", "")
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = recipient_email

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, clean_pass)
            server.sendmail(sender_email, recipient_email, msg.as_string())
        return True, "Email alert sent successfully!"
    except Exception as e:
        return False, f"Failed to send email alert: {e}"


def evaluate_and_dispatch_alerts(curr_temp, rain_prob, target_city_name, settings):
    alerts = []
    if curr_temp >= settings["max_temp"]:
        alerts.append(
            f"🔥 Extreme Heat Warning: Temperature reached {curr_temp}°C (Threshold: {settings['max_temp']}°C)")
    if rain_prob >= settings["rain_prob"]:
        alerts.append(
            f"🌧️ Heavy Rain Warning: Chance of rain today is {rain_prob}% (Threshold: {settings['rain_prob']}%)")

    if alerts:
        alert_body = f"⚠️ WEATHER ALERT FOR {target_city_name.upper()} ⚠️\n\n" + "\n".join(alerts)

        st.warning(f"🚨 **{len(alerts)} Weather Alert(s) Triggered!**")
        for alert in alerts:
            st.error(alert)

        if settings.get("enable_whatsapp"):
            success, msg = send_whatsapp_alert(settings["wa_sid"], settings["wa_token"], settings["wa_from"],
                                               settings["wa_to"], alert_body)
            if success:
                st.toast("💬 WhatsApp alert sent!", icon="✅")
            else:
                st.sidebar.error(msg)

        if settings.get("enable_email"):
            success, msg = send_email_alert(settings["sender_email"], settings["sender_pass"],
                                            settings["recipient_email"], f"Weather Alert: {target_city_name}",
                                            alert_body)
            if success:
                st.toast("📧 Email alert sent!", icon="✅")
            else:
                st.sidebar.error(msg)


# --- STREAMLIT UI ---
st.title("🌤️ Weather Forecast & Climate Intelligence")

# Sidebar Configuration
st.sidebar.header("⚙️ Dashboard Controls")
input_city = st.sidebar.text_input("Enter City Name:", value="Karachi")
enable_audio = st.sidebar.checkbox("Enable Voice Audio Output", value=True)

st.sidebar.markdown("---")
st.sidebar.header("🔔 Smart Alert Triggers")
enable_alerts = st.sidebar.checkbox("Enable Automated Alerting System", value=False)

alert_settings = {}
if enable_alerts:
    st.sidebar.subheader("Threshold Settings")
    alert_settings["max_temp"] = st.sidebar.slider("Trigger Alert Temp (°C) ≥", 30, 50, 40)
    alert_settings["rain_prob"] = st.sidebar.slider("Trigger Rain Prob (%) ≥", 40, 100, 70)

    st.sidebar.subheader("Notification Channels")
    alert_settings["enable_whatsapp"] = st.sidebar.checkbox("WhatsApp Alerts", value=False)
    if alert_settings["enable_whatsapp"]:
        alert_settings["wa_sid"] = st.sidebar.text_input("Twilio Account SID",
                                                         value=os.getenv("TWILIO_ACCOUNT_SID", ""), type="password")
        alert_settings["wa_token"] = st.sidebar.text_input("Twilio Auth Token",
                                                           value=os.getenv("TWILIO_AUTH_TOKEN", ""), type="password")
        alert_settings["wa_from"] = st.sidebar.text_input("Twilio WhatsApp No.",
                                                          value=os.getenv("TWILIO_WHATSAPP_NUMBER", "+14155238886"))
        alert_settings["wa_to"] = st.sidebar.text_input("Recipient WhatsApp No.",
                                                        value=os.getenv("RECIPIENT_WHATSAPP_NUMBER", ""))

    alert_settings["enable_email"] = st.sidebar.checkbox("Email Alerts", value=True)
    if alert_settings["enable_email"]:
        alert_settings["sender_email"] = st.sidebar.text_input("Sender Email (Gmail)",
                                                               value=os.getenv("SENDER_EMAIL", ""))
        alert_settings["sender_pass"] = st.sidebar.text_input("App Password", value=os.getenv("SENDER_PASS", ""),
                                                              type="password")
        alert_settings["recipient_email"] = st.sidebar.text_input("Recipient Email",
                                                                  value=os.getenv("RECIPIENT_EMAIL", ""))

# Fetch City Coordinates
lat, lon, city_name, country, geo_err = get_city_coordinates(input_city)

if geo_err:
    st.error(f"[Error] {geo_err}")
else:
    # Fetch 16-Day Forecast & Real-Time Data
    forecast_data, fc_err = get_current_and_16day_forecast(lat, lon)

    if fc_err:
        st.error(f"[Error] {fc_err}")
    elif forecast_data:
        current_data = forecast_data["current"]
        daily_data = forecast_data["daily"]

        temp = current_data["temperature_2m"]
        feels_like = current_data["apparent_temperature"]
        humidity = current_data["relative_humidity_2m"]
        wind = current_data["wind_speed_10m"]

        # Parse Local Time & Timezone Abbreviation
        raw_iso_time = current_data.get("time", "")
        tz_abbr = forecast_data.get("timezone_abbreviation", "Local")

        if raw_iso_time:
            dt_obj = datetime.datetime.fromisoformat(raw_iso_time)
            formatted_time = dt_obj.strftime("%A, %b %d, %Y | %I:%M %p")
        else:
            formatted_time = "N/A"

        # Header Info Banner (Custom CSS Card with Local Time)
        st.markdown(f"""
        <div style="background: rgba(28, 30, 42, 0.75); border: 1px solid rgba(255,255,255,0.08); padding: 20px; border-radius: 20px; margin-bottom: 25px; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="margin: 0; font-size: 2.2rem; color: #ffffff;">📍 {city_name}, {country}</h1>
                <div style="color: #00cec9; font-size: 0.95rem; font-weight: 600; margin-top: 4px;">
                    🕒 Local Time: {formatted_time} ({tz_abbr})
                </div>
                <span style="color: #a0a5b5; font-size: 0.85rem;">Coordinates: {lat:.2f}°N, {lon:.2f}°E | Live Telemetry Active</span>
            </div>
            <div style="text-align: right;">
                <span style="color: #8c93a8; font-size: 0.85rem; text-transform: uppercase;">Live Temp</span>
                <div style="color: #ff9f43; font-size: 2.2rem; font-weight: 700;">{temp}°C</div>
                <span style="color: #00cec9; font-size: 0.85rem;">Feels {feels_like}°C</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Main Dashboard Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Current & 16-Day Forecast",
            "📜 Historical Trends (5 Years)",
            "🔮 ML Projections (Next 5 Years)",
            "🚨 Smart Alerts Engine",
            "📥 Export Data"
        ])

        # TAB 1: CURRENT METRICS, LIFESTYLE INDEX & 16-DAY FORECAST GRID
        with tab1:
            st.subheader("Real-Time Conditions")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div style="background: rgba(28, 30, 42, 0.75); border: 1px solid rgba(255,255,255,0.08); padding: 18px; border-radius: 18px;">
                    <span style="color: #8c93a8; font-size: 0.9rem;">Temperature</span>
                    <h2 style="color: #ffffff; margin: 6px 0; font-size: 2.2rem;">{temp}°C</h2>
                    <span style="color: #ff9f43; font-size: 0.85rem;">Feels like {feels_like}°C</span>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div style="background: rgba(28, 30, 42, 0.75); border: 1px solid rgba(255,255,255,0.08); padding: 18px; border-radius: 18px;">
                    <span style="color: #8c93a8; font-size: 0.9rem;">Relative Humidity</span>
                    <h2 style="color: #ffffff; margin: 6px 0; font-size: 2.2rem;">{humidity}%</h2>
                    <span style="color: #00cec9; font-size: 0.85rem;">Optimal Moisture</span>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div style="background: rgba(28, 30, 42, 0.75); border: 1px solid rgba(255,255,255,0.08); padding: 18px; border-radius: 18px;">
                    <span style="color: #8c93a8; font-size: 0.9rem;">Wind Speed</span>
                    <h2 style="color: #ffffff; margin: 6px 0; font-size: 2.2rem;">{wind} <span style="font-size: 1rem;">km/h</span></h2>
                    <span style="color: #54a0ff; font-size: 0.85rem;">Live Telemetry</span>
                </div>
                """, unsafe_allow_html=True)

            # LIFESTYLE & OUTDOOR ACTIVITY SCORE INDEX
            dates = daily_data["time"]
            max_temps = daily_data["temperature_2m_max"]
            min_temps = daily_data["temperature_2m_min"]
            rain_probs = daily_data["precipitation_probability_max"]

            today_rain = rain_probs[0] if rain_probs and rain_probs[0] is not None else 0
            lifestyle_scores = calculate_lifestyle_scores(temp, humidity, wind, today_rain)

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("🏃 Outdoor Lifestyle & Activity Index")

            col_sp, col_ld, col_cm, col_tr = st.columns(4)
            metrics_display = [
                (col_sp, "🏃 Outdoor Sports", lifestyle_scores["Sports"]),
                (col_ld, "👕 Laundry Drying", lifestyle_scores["Laundry"]),
                (col_cm, "🚗 Commute Safety", lifestyle_scores["Commuting"]),
                (col_tr, "🧳 Travel & Sightseeing", lifestyle_scores["Travel"])
            ]

            for ui_col, title, score_val in metrics_display:
                status_color, status_label = get_score_status(score_val)
                with ui_col:
                    st.markdown(f"""
                    <div style="background: rgba(28, 30, 42, 0.85); border: 1px solid rgba(255, 255, 255, 0.08); padding: 16px; border-radius: 18px; text-align: center;">
                        <div style="color: #a0a5b5; font-size: 0.9rem; font-weight: 600;">{title}</div>
                        <div style="color: {status_color}; font-size: 2.2rem; font-weight: 800; margin: 8px 0;">{score_val}<span style="font-size: 1rem; color: #a0a5b5;">/100</span></div>
                        <div style="background: {status_color}22; color: {status_color}; border: 1px solid {status_color}44; border-radius: 12px; padding: 4px 10px; font-size: 0.8rem; font-weight: 600; display: inline-block;">
                            {status_label}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader(f"📅 16-Day Short-Term Forecast")

            grid_cols = st.columns(3)
            for i in range(len(dates)):
                rain_val = rain_probs[i] if rain_probs[i] is not None else 0
                with grid_cols[i % 3]:
                    st.markdown(f"""
                    <div style="background: rgba(28, 30, 42, 0.85); border: 1px solid rgba(255, 255, 255, 0.08); padding: 16px; border-radius: 18px; margin-bottom: 15px;">
                        <div style="color: #a0a5b5; font-size: 0.85rem; margin-bottom: 6px;">📅 {dates[i]}</div>
                        <div style="color: #8c93a8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.5px;">High / Low</div>
                        <div style="display: flex; align-items: baseline; gap: 10px; margin: 6px 0;">
                            <span style="color: #ffffff; font-size: 1.8rem; font-weight: 700;">{max_temps[i]}°C</span>
                            <span style="color: #ff9f43; font-size: 0.95rem; font-weight: 600;">↓ {min_temps[i]}°C</span>
                        </div>
                        <div style="color: #a0a5b5; font-size: 0.85rem; margin-top: 8px;">🌧️ Rain: {rain_val}%</div>
                        <div style="background: rgba(255,255,255,0.1); border-radius: 6px; height: 6px; width: 100%; margin-top: 6px; overflow: hidden;">
                            <div style="background: #ff9f43; width: {rain_val}%; height: 100%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # Load 5-Year History and ML Data
        with st.status(f"Fetching climate archives & training ML algorithms for {city_name}...",
                       expanded=False) as status:
            st.write("📡 Connecting to Open-Meteo Archives...")
            df_hist_daily, hist_monthly, future_monthly, df_future_daily, ml_err = fetch_history_and_forecast_5years(
                lat, lon)
            st.write("🤖 Fitting Random Forest Regressors...")
            st.write("📈 Generating Future Climate Projections...")
            status.update(label="Climate Analysis & ML Training Completed!", state="complete", expanded=False)

        contrast_palette = ["#FF9F43", "#00CEC9", "#FF6B6B", "#FECA57", "#54A0FF"]

        # TAB 2: LAST 5 YEARS HISTORICAL WEATHER
        with tab2:
            st.subheader(f"📜 Historical Climate Trends for {city_name} (Last 5 Years)")
            if ml_err:
                st.error(ml_err)
            elif hist_monthly is not None:
                selected_year_hist = st.selectbox(
                    "Select Historical Year Filter:",
                    options=["All Years"] + sorted(list(hist_monthly["year"].unique()), reverse=True)
                )

                if selected_year_hist != "All Years":
                    filtered_hist = hist_monthly[hist_monthly["year"] == selected_year_hist]
                else:
                    filtered_hist = hist_monthly

                fig_hist = px.line(
                    filtered_hist,
                    x="month_name",
                    y=["Avg_Max_Temp", "Avg_Min_Temp"],
                    color="year",
                    title=f"Historical Monthly Temperature Trends (°C)",
                    labels={"month_name": "Month", "value": "Temperature (°C)", "variable": "Metric"},
                    markers=True,
                    line_shape="spline",
                    color_discrete_sequence=contrast_palette
                )
                fig_hist.update_layout(
                    template="plotly_dark",
                    hovermode="x unified",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#A0A5B5", family="sans-serif")
                )
                st.plotly_chart(fig_hist, use_container_width=True)

                st.markdown("### 📊 Historical Data Summary")
                formatted_hist = filtered_hist[
                    ["year", "month_name", "Avg_Max_Temp", "Avg_Min_Temp", "Total_Precipitation"]].copy()
                formatted_hist.columns = ["Year", "Month", "Avg Max Temp", "Avg Min Temp", "Total Rain"]

                st.dataframe(
                    formatted_hist,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Avg Max Temp": st.column_config.NumberColumn("Avg Max Temp (°C)", format="%.1f °C"),
                        "Avg Min Temp": st.column_config.NumberColumn("Avg Min Temp (°C)", format="%.1f °C"),
                        "Total Rain": st.column_config.NumberColumn("Total Rain (mm)", format="%.1f mm"),
                    }
                )

        # TAB 3: NEXT 5 YEARS ML WEATHER FORECAST
        with tab3:
            st.subheader(f"🔮 Month-Wise 5-Year ML Climate Projections")
            if ml_err:
                st.error(ml_err)
            elif future_monthly is not None:
                selected_year_fut = st.selectbox(
                    "Select Future Projection Year Filter:",
                    options=["All Future Years"] + sorted(list(future_monthly["year"].unique()))
                )

                if selected_year_fut != "All Future Years":
                    filtered_future = future_monthly[future_monthly["year"] == selected_year_fut]
                else:
                    filtered_future = future_monthly

                fig_fut = px.line(
                    filtered_future,
                    x="month_name",
                    y=["Predicted_Avg_Max_Temp", "Predicted_Avg_Min_Temp"],
                    color="year",
                    title=f"ML Predicted Monthly Temperature (°C)",
                    labels={"month_name": "Month", "value": "Temperature (°C)", "variable": "Metric"},
                    markers=True,
                    line_shape="spline",
                    color_discrete_sequence=contrast_palette
                )
                fig_fut.update_layout(
                    template="plotly_dark",
                    hovermode="x unified",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#A0A5B5", family="sans-serif")
                )
                st.plotly_chart(fig_fut, use_container_width=True)

                st.markdown("### 📊 Projected Data Summary")
                formatted_future = filtered_future[
                    ["year", "month_name", "Predicted_Avg_Max_Temp", "Predicted_Avg_Min_Temp"]].copy()
                formatted_future.columns = ["Year", "Month", "Predicted Avg Max", "Predicted Avg Min"]

                st.dataframe(
                    formatted_future,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Predicted Avg Max": st.column_config.NumberColumn("Predicted Max (°C)", format="%.1f °C"),
                        "Predicted Avg Min": st.column_config.NumberColumn("Predicted Min (°C)", format="%.1f °C"),
                    }
                )

        # TAB 4: ALERTS SYSTEM
        with tab4:
            st.subheader("Smart Threshold Monitoring")
            if enable_alerts:
                today_rain_prob = rain_probs[0] if rain_probs and rain_probs[0] is not None else 0
                evaluate_and_dispatch_alerts(temp, today_rain_prob, city_name, alert_settings)
            else:
                st.info("Enable Automated Alerting System in the sidebar to configure triggers.")

        # TAB 5: DATA EXPORT MODULE
        with tab5:
            st.subheader("📥 Export Weather Data & Reports")
            st.caption(
                "Filter historical archives or future machine learning projections for offline scientific analysis.")

            if ml_err or hist_monthly is None or df_hist_daily is None or future_monthly is None or df_future_daily is None:
                st.error("Cannot export datasets due to a data loading error.")
            else:
                export_category = st.selectbox(
                    "Choose Dataset Category to Export:",
                    options=[
                        "Historical Monthly Trends (Last 5 Years)",
                        "Historical Daily Granular Data (Last 5 Years)",
                        "Predicted Monthly Climate Projections (Next 5 Years)",
                        "Predicted Daily Granular Projections (Next 5 Years)"
                    ]
                )

                if export_category == "Historical Monthly Trends (Last 5 Years)":
                    export_df = hist_monthly.copy()
                    filename_prefix = f"{city_name}_historical_monthly"
                elif export_category == "Historical Daily Granular Data (Last 5 Years)":
                    export_df = df_hist_daily.copy()
                    filename_prefix = f"{city_name}_historical_daily"
                elif export_category == "Predicted Monthly Climate Projections (Next 5 Years)":
                    export_df = future_monthly.copy()
                    filename_prefix = f"{city_name}_predicted_monthly"
                else:
                    export_df = df_future_daily.copy()
                    filename_prefix = f"{city_name}_predicted_daily"

                st.markdown(f"**Dataset Preview ({len(export_df)} records)**")
                st.dataframe(export_df.head(10), use_container_width=True, hide_index=True)

                col_csv, col_pdf = st.columns(2)

                with col_csv:
                    csv_bytes = convert_df_to_csv(export_df)
                    st.download_button(
                        label="📄 Download Dataset (CSV)",
                        data=csv_bytes,
                        file_name=f"{filename_prefix}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                with col_pdf:
                    pdf_bytes = generate_pdf_report(export_df, city_name, export_category)
                    st.download_button(
                        label="📕 Download Summary Report (PDF)",
                        data=pdf_bytes,
                        file_name=f"{filename_prefix}_report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

        # TRIGGER AUDIO OUTPUT
        if enable_audio:
            summary_speech = f"Current temperature in {city_name} is {temp} degrees Celsius with humidity at {humidity} percent."
            speak(summary_speech)