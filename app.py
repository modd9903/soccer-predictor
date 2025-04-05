import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
from xgboost import XGBClassifier

# --- Title ---
st.title("⚽ Soccer Match Predictor with Live Stats (Understat Autofill)")

if 'history' not in st.session_state:
    st.session_state.history = []

@st.cache_resource
def load_model():
    np.random.seed(42)
    X_dummy = np.random.normal(loc=0, scale=1, size=(100, 11))
    y_dummy = np.random.choice([0, 1, 2], 100)
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_dummy, y_dummy)
    return model

clf = load_model()

# --- Autofill Section ---
st.header("Paste Understat Match Link (optional)")
understat_url = st.text_input("Understat Match URL (e.g. https://understat.com/match/20530)")

auto_data = {}
if understat_url and "understat.com/match" in understat_url:
    try:
        page = requests.get(understat_url)
        soup = BeautifulSoup(page.content, "html.parser")
        script = soup.find_all("script")
        for s in script:
            if "match_info" in s.text:
                json_data = s.text.split("JSON.parse('")[1].split("')")[0]
                json_data = json.loads(json_data.encode('utf-8').decode('unicode_escape'))
                break

        team1 = json_data['h']['title']
        team2 = json_data['a']['title']
        team1_xg = float(json_data['h']['xG']['sum'])
        team2_xg = float(json_data['a']['xG']['sum'])
        team1_goals = int(json_data['h']['goals']['sum'])
        team2_goals = int(json_data['a']['goals']['sum'])
        # Possession is not on Understat, fake it equally
        auto_data = {
            'home_team': team1,
            'away_team': team2,
            'home_xg': team1_xg,
            'away_xg': team2_xg,
            'home_goals': team1_goals,
            'away_goals': team2_goals,
            'home_possession': 50,
            'away_possession': 50,
        }
        st.success(f"Auto-filled stats loaded from Understat for {team1} vs {team2}")
    except Exception as e:
        st.error("Failed to fetch data from Understat.")

# --- Team Input ---
st.header("Team Info")
home_team = st.text_input("Home Team", value=auto_data.get('home_team', ''))
away_team = st.text_input("Away Team", value=auto_data.get('away_team', ''))

# --- Team Stats ---
st.header("Team Performance Stats")
home_goals_scored = st.number_input("Home Avg Goals Scored", min_value=0.0, value=auto_data.get('home_goals', 0.0))
home_goals_conceded = st.number_input("Home Avg Goals Conceded", min_value=0.0)
away_goals_scored = st.number_input("Away Avg Goals Scored", min_value=0.0, value=auto_data.get('away_goals', 0.0))
away_goals_conceded = st.number_input("Away Avg Goals Conceded", min_value=0.0)

home_possession = st.slider("Home Possession (%)", 0, 100, auto_data.get('home_possession', 50))
away_possession = st.slider("Away Possession (%)", 0, 100, auto_data.get('away_possession', 50))
home_shot_accuracy = st.slider("Home Shot Accuracy (%)", 0, 100, 50)
away_shot_accuracy = st.slider("Away Shot Accuracy (%)", 0, 100, 50)

home_xg = st.number_input("Home Expected Goals (xG)", min_value=0.0, value=auto_data.get('home_xg', 0.0))
home_xga = st.number_input("Home Expected Goals Against (xGA)", min_value=0.0)
away_xg = st.number_input("Away Expected Goals (xG)", min_value=0.0, value=auto_data.get('away_xg', 0.0))
away_xga = st.number_input("Away Expected Goals Against (xGA)", min_value=0.0)

# --- Form & Other Inputs ---
st.header("Recent Form (last 5 games)")
home_form = st.text_input("Home Form (e.g. W-W-L-D-W)")
away_form = st.text_input("Away Form")

importance = st.selectbox("Match Importance", ["Normal", "Rivalry", "Cup Final"])
importance_score = {"Normal": 0, "Rivalry": 1, "Cup Final": 2}[importance]

home_missing = st.text_area("Home Missing Players (name:importance)")
away_missing = st.text_area("Away Missing Players (name:importance)")

home_stars = st.text_area("Home Star Players (name:rating)")
away_stars = st.text_area("Away Star Players (name:rating)")

h2h_results = st.text_area("Head to Head Results (one per line, e.g. Arsenal 2-1 Liverpool)")

st.header("Intuition Boost")
intuition_boost = st.slider("Your gut feeling (favor home -1.0 to 1.0 favor away)", -1.0, 1.0, 0.0, step=0.1)

actual_result = st.selectbox("Actual Result (if known)", ["--", "Home Win", "Draw", "Away Win"])

if st.button("Predict Match Result"):
    def parse_form(form):
        scores = {'W': 3, 'D': 1, 'L': 0}
        return sum([scores.get(x.strip(), 0) for x in form.split('-')]) / (len(form.split('-')) * 3)

    def parse_players(data):
        total = 0
        for line in data.strip().split('\n'):
            try:
                name, value = line.split(':')
                total += float(value.strip())
            except:
                pass
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
        total_games = home_wins + away_wins + draws
        if total_games == 0:
            return 0.5
        return (home_wins * 3 + draws) / (total_games * 3)

    # Feature extraction
    form_score_home = parse_form(home_form)
    form_score_away = parse_form(away_form)
    goals_score_home = home_goals_scored - home_goals_conceded
    goals_score_away = away_goals_scored - away_goals_conceded
    missing_penalty_home = parse_players(home_missing)
    missing_penalty_away = parse_players(away_missing)
    star_power_home = parse_players(home_stars)
    star_power_away = parse_players(away_stars)
    h2h_score = parse_h2h(h2h_results)

    features = np.array([[form_score_home, form_score_away,
                          goals_score_home, goals_score_away,
                          star_power_home - missing_penalty_home,
                          star_power_away - missing_penalty_away,
                          h2h_score, intuition_boost,
                          (home_possession - away_possession) / 100,
                          (home_shot_accuracy - away_shot_accuracy) / 100,
                          importance_score]])

    pred = clf.predict(features)[0]
    prob = clf.predict_proba(features)[0]
    result_label = ["Draw", f"{home_team} Win", f"{away_team} Win"]
    original_result = result_label[pred]
    confidence = round(np.max(prob) * 100, 1)

    net_strength_home = goals_score_home + (star_power_home - missing_penalty_home) / 10 + form_score_home
    net_strength_away = goals_score_away + (star_power_away - missing_penalty_away) / 10 + form_score_away
    total_strength = net_strength_home + net_strength_away + 0.01
    expected_goals_home = round(max(0, 2.5 * (net_strength_home / total_strength)), 1)
    expected_goals_away = round(max(0, 2.5 * (net_strength_away / total_strength)), 1)

    if abs(expected_goals_home - expected_goals_away) <= 0.4:
        result = "Draw"
    elif expected_goals_home > expected_goals_away:
        result = f"{home_team} Win"
    else:
        result = f"{away_team} Win"

    # Display
    st.subheader("🏁 Prediction Result (Live + ML + Intuition)")
    st.write(f"**Predicted Result:** {result}")
    st.write(f"**Confidence:** {confidence:.1f}%")
    st.write(f"**Predicted Scoreline:** {home_team} {expected_goals_home} - {expected_goals_away} {away_team}")
    st.write(f"**Raw ML Prediction:** {original_result}")
