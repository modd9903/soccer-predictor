import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import json
from xgboost import XGBClassifier

# --- Title ---
st.title("⚽ Soccer Match Predictor (XGBoost + Intuition + Accuracy Tracking + Understat Support)")

# --- Session State ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- Load Dummy Model ---
@st.cache_resource
def load_model():
    X_dummy = np.random.normal(loc=0, scale=1, size=(100, 11))
    y_dummy = np.random.choice([0, 1, 2], 100)
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_dummy, y_dummy)
    return model

clf = load_model()

# --- Understat Fetch ---
def fetch_understat_data(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, "html.parser")
        script = soup.find("script", text=lambda t: t and "shotsData" in t)
        if not script:
            return None
        text = script.string
        start = text.find("shotsData") + len("shotsData = ")
        end = text.find("];", start) + 1
        json_data = text[start:end]
        return json.loads(json_data)
    except:
        return None

# --- Understat Input ---
st.subheader("📥 Paste Understat Match Link (optional)")
understat_url = st.text_input("Understat Match URL (e.g. https://understat.com/match/20530)")

# Default values
home_team = away_team = ""
home_goals_scored = home_goals_conceded = away_goals_scored = away_goals_conceded = 1.0
home_possession = away_possession = 50
home_shot_accuracy = away_shot_accuracy = 50
home_xg = home_xga = away_xg = away_xga = 1.0

# Try auto-population
if understat_url:
    data = fetch_understat_data(understat_url)
    if data and len(data) >= 2:
        home = data[0]
        away = data[1]
        home_team = home.get("h_team", "")
        away_team = home.get("a_team", "")
        home_goals_scored = float(home.get("result", "1"))
        away_goals_scored = float(away.get("result", "1"))
        home_xg = float(home.get("xG", 1.0))
        away_xg = float(away.get("xG", 1.0))
        # Dummy values if not available
        st.success("Auto-fetched data from Understat.")
    else:
        st.error("Failed to fetch data from Understat.")

# --- Team Inputs ---
st.header("Team Info")
home_team = st.text_input("Home Team", home_team)
away_team = st.text_input("Away Team", away_team)

# --- Stats ---
st.header("Team Performance Stats")
home_goals_scored = st.number_input("Home Avg Goals Scored", value=home_goals_scored)
home_goals_conceded = st.number_input("Home Avg Goals Conceded", value=home_goals_conceded)
away_goals_scored = st.number_input("Away Avg Goals Scored", value=away_goals_scored)
away_goals_conceded = st.number_input("Away Avg Goals Conceded", value=away_goals_conceded)

home_possession = st.slider("Home Possession (%)", 0, 100, home_possession)
away_possession = st.slider("Away Possession (%)", 0, 100, away_possession)
home_shot_accuracy = st.slider("Home Shot Accuracy (%)", 0, 100, home_shot_accuracy)
away_shot_accuracy = st.slider("Away Shot Accuracy (%)", 0, 100, away_shot_accuracy)

home_xg = st.number_input("Home xG", value=home_xg)
home_xga = st.number_input("Home xGA", value=home_xga)
away_xg = st.number_input("Away xG", value=away_xg)
away_xga = st.number_input("Away xGA", value=away_xga)

# --- Form ---
st.header("Form")
home_form = st.text_input("Home Form (W-W-D-L-W)")
away_form = st.text_input("Away Form")

# --- Players ---
st.header("Missing/Star Players")
home_missing = st.text_area("Home Missing (name:importance)")
away_missing = st.text_area("Away Missing (name:importance)")
home_stars = st.text_area("Home Star Players (name:rating)")
away_stars = st.text_area("Away Star Players (name:rating)")

# --- H2H ---
st.header("Head-to-Head")
h2h_results = st.text_area("Last 5 games (e.g. Arsenal 2-1 Liverpool)")

# --- Intuition ---
st.header("Intuition Boost")
intuition_boost = st.slider("Favor home (-1.0) to away (+1.0)", -1.0, 1.0, 0.0, 0.1)

# --- Match Importance ---
importance = st.selectbox("Match Importance", ["Normal", "Rivalry", "Cup Final"])
importance_score = {"Normal": 0, "Rivalry": 1, "Cup Final": 2}[importance]

# --- Optional Actual Result Entry ---
actual_result = st.selectbox("Actual Result (if known)", ["--", "Home Win", "Draw", "Away Win"])

# --- Parse Functions ---
def parse_form(form):
    scores = {'W': 3, 'D': 1, 'L': 0}
    return sum([scores.get(x.strip().upper(), 0) for x in form.split('-')]) / (len(form.split('-')) * 3) if form else 0.5

def parse_players(data):
    total = 0
    for line in data.strip().split('\n'):
        try:
            name, value = line.split(':')
            total += float(value.strip())
        except:
            continue
    return total

def parse_h2h(h2h):
    home_wins = away_wins = draws = 0
    lines = h2h.strip().split('\n')
    for line in lines:
        try:
            if home_team in line and away_team in line:
                parts = line.split()
                scores = [int(s) for part in parts for s in part.split('-') if s.isdigit()]
                if len(scores) < 2:
                    continue
                if line.startswith(home_team):
                    home_score, away_score = scores[0], scores[1]
                else:
                    away_score, home_score = scores[0], scores[1]
                if home_score > away_score:
                    home_wins += 1
                elif home_score < away_score:
                    away_wins += 1
                else:
                    draws += 1
        except:
            continue
    total = home_wins + away_wins + draws
    return (home_wins * 3 + draws) / (total * 3) if total else 0.5

# --- Predict ---
if st.button("Predict Match Result"):
    form_score_home = parse_form(home_form)
    form_score_away = parse_form(away_form)
    goals_score_home = home_goals_scored - home_goals_conceded
    goals_score_away = away_goals_scored - away_goals_conceded
    missing_home = parse_players(home_missing)
    missing_away = parse_players(away_missing)
    stars_home = parse_players(home_stars)
    stars_away = parse_players(away_stars)
    h2h_score = parse_h2h(h2h_results)

    features = np.array([[form_score_home, form_score_away,
                          goals_score_home, goals_score_away,
                          stars_home - missing_home,
                          stars_away - missing_away,
                          h2h_score, intuition_boost,
                          (home_possession - away_possession) / 100,
                          (home_shot_accuracy - away_shot_accuracy) / 100,
                          importance_score]])

    pred = clf.predict(features)[0]
    prob = clf.predict_proba(features)[0]
    result_label = ["Draw", f"{home_team} Win", f"{away_team} Win"]
    result = result_label[pred]
    raw_result = result

    # Scoreline logic
    net_home = goals_score_home + (stars_home - missing_home) / 10 + form_score_home
    net_away = goals_score_away + (stars_away - missing_away) / 10 + form_score_away
    total = net_home + net_away + 0.01
    score_home = round(max(0, 2.5 * (net_home / total)), 1)
    score_away = round(max(0, 2.5 * (net_away / total)), 1)

    if abs(score_home - score_away) <= 0.4:
        result = "Draw"
    elif score_home > score_away:
        result = f"{home_team} Win"
    else:
        result = f"{away_team} Win"

    confidence = round(np.max(prob) * 100, 1)

    # --- Output ---
    st.subheader("🏁 Prediction Result (ML + Intuition)")
    st.write(f"**Predicted Result:** {result}")
    st.write(f"**Confidence Level:** {confidence:.1f}%")
    st.write(f"**Predicted Scoreline:** {home_team} {score_home} - {score_away} {away_team}")
    st.write(f"**ML Model Prediction (Raw):** {raw_result}")

    # --- Chart ---
    st.subheader("🔍 Win Probability Distribution")
    fig, ax = plt.subplots()
    ax.bar(["Draw", f"{home_team} Win", f"{away_team} Win"], prob, color=["gray", "green", "red"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    st.pyplot(fig)

    # --- History ---
    outcome_map = {"Home Win": f"{home_team} Win", "Draw": "Draw", "Away Win": f"{away_team} Win"}
    correct = "✅" if (actual_result != "--" and result == outcome_map.get(actual_result)) else "❌" if actual_result != "--" else "N/A"

    st.session_state.history.append({
        "Match": f"{home_team} vs {away_team}",
        "Prediction": result,
        "Confidence": f"{confidence:.1f}%",
        "Actual": actual_result if actual_result != "--" else "(Not entered)",
        "Correct": correct
    })

    if st.session_state.history:
        st.subheader("📊 Prediction History")
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)
        correct_preds = df[df["Correct"] == "✅"].shape[0]
        total_preds = df[df["Correct"].isin(["✅", "❌"])].shape[0]
        if total_preds:
            st.success(f"Accuracy: {correct_preds}/{total_preds} ({correct_preds / total_preds * 100:.1f}%)")
