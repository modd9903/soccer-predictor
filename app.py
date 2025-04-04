import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- Title ---
st.title("⚽ Soccer Match Predictor (ML + Intuition)")

# --- Team Input ---
st.header("Team Info")
home_team = st.text_input("Home Team")
away_team = st.text_input("Away Team")

# --- Recent Form ---
st.header("Recent Form (last 5 games)")
home_form = st.text_input("Home Form (e.g. W-W-L-D-W)")
away_form = st.text_input("Away Form")

# --- Team Stats ---
st.header("Team Performance Stats")
home_goals_scored = st.number_input("Home Avg Goals Scored", min_value=0.0)
home_goals_conceded = st.number_input("Home Avg Goals Conceded", min_value=0.0)
away_goals_scored = st.number_input("Away Avg Goals Scored", min_value=0.0)
away_goals_conceded = st.number_input("Away Avg Goals Conceded", min_value=0.0)

# --- Key Players Missing ---
st.header("Missing Players")
home_missing = st.text_area("Home Missing Players (name:importance, one per line)")
away_missing = st.text_area("Away Missing Players (name:importance)")

# --- Star Players ---
st.header("Star Players Available")
home_stars = st.text_area("Home Star Players (name:rating)")
away_stars = st.text_area("Away Star Players (name:rating)")

# --- Head to Head ---
st.header("Head to Head Results (last 5 games)")
h2h_results = st.text_area("One per line (e.g. Arsenal 2-1 Liverpool)")

# --- Intuition Boost ---
st.header("Intuition Boost")
intuition_boost = st.slider("Your gut feeling (favor home -1.0 to 1.0 favor away)", -1.0, 1.0, 0.0, step=0.1)

if st.button("Predict Match Result"):
    # --- Helper Functions ---
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

    # --- Extract Inputs ---
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
                          h2h_score, intuition_boost]])

    # --- Dummy Training Data (mocked) ---
    np.random.seed(42)
    X_dummy = np.random.rand(300, 8) * 2 - 1
    y_dummy = np.random.choice([0, 1, 2], 300)  # 0 = Draw, 1 = Home Win, 2 = Away Win

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_dummy, y_dummy)

    pred = clf.predict(features)[0]
    prob = clf.predict_proba(features)[0]

    result_label = ["Draw", f"{home_team} Win", f"{away_team} Win"]
    result = result_label[pred]
    confidence = round(np.max(prob) * 100, 1)

    # --- Display Result ---
    st.subheader("🏁 Prediction Result (ML + Intuition)")
    st.write(f"**Predicted Result:** {result}")
    st.write(f"**Confidence Level:** {confidence}%")

    # --- Score Estimate (same as before) ---
    predicted_home_goals = round(home_goals_scored * (1 - missing_penalty_home / 20), 1)
    predicted_away_goals = round(away_goals_scored * (1 - missing_penalty_away / 20), 1)
    st.write(f"**Predicted Scoreline:** {home_team} {predicted_home_goals} - {predicted_away_goals} {away_team}")
