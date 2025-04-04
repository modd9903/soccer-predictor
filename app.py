import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# --- Title ---
st.title("⚽ Soccer Match Predictor (XGBoost + Intuition + Accuracy Tracking)")

# --- Session State for Logging ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- Cached Model (Faster) ---
@st.cache_resource
def load_model():
    np.random.seed(42)
    X_dummy = np.random.normal(loc=0, scale=1, size=(100, 11))
    y_dummy = np.random.choice([0, 1, 2], 100)
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_dummy, y_dummy)
    return model

clf = load_model()

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

home_possession = st.slider("Home Possession (%)", 0, 100, 50)
away_possession = st.slider("Away Possession (%)", 0, 100, 50)
home_shot_accuracy = st.slider("Home Shot Accuracy (%)", 0, 100, 50)
away_shot_accuracy = st.slider("Away Shot Accuracy (%)", 0, 100, 50)

home_xg = st.number_input("Home Expected Goals (xG)", min_value=0.0)
home_xga = st.number_input("Home Expected Goals Against (xGA)", min_value=0.0)
away_xg = st.number_input("Away Expected Goals (xG)", min_value=0.0)
away_xga = st.number_input("Away Expected Goals Against (xGA)", min_value=0.0)

importance = st.selectbox("Match Importance", ["Normal", "Rivalry", "Cup Final"])
importance_score = {"Normal": 0, "Rivalry": 1, "Cup Final": 2}[importance]

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

# --- Optional Actual Result Entry ---
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
                          h2h_score, intuition_boost,
                          (home_possession - away_possession) / 100,
                          (home_shot_accuracy - away_shot_accuracy) / 100,
                          importance_score]])

    pred = clf.predict(features)[0]
    prob = clf.predict_proba(features)[0]

    result_label = ["Draw", f"{home_team} Win", f"{away_team} Win"]
    result = result_label[pred]
    confidence = round(np.max(prob) * 100, 1)

    net_strength_home = goals_score_home + (star_power_home - missing_penalty_home) / 10 + form_score_home
    net_strength_away = goals_score_away + (star_power_away - missing_penalty_away) / 10 + form_score_away
    total_strength = net_strength_home + net_strength_away + 0.01
    expected_goals_home = round(2.5 * (net_strength_home / total_strength), 1)
    expected_goals_away = round(2.5 * (net_strength_away / total_strength), 1)

    if abs(expected_goals_home - expected_goals_away) <= 0.4:
        result = "Draw"

    st.subheader("\U0001F3C1 Prediction Result (ML + Intuition)")
    st.write(f"**Predicted Result:** {result}")
    st.write(f"**Confidence Level:** {confidence:.1f}%")
    st.write(f"**Predicted Scoreline:** {home_team} {expected_goals_home} - {expected_goals_away} {away_team}")

    st.subheader("🔍 Win Probability Distribution")
    fig, ax = plt.subplots()
    ax.bar([f"{home_team} Win", "Draw", f"{away_team} Win"], prob, color=['green', 'gray', 'red'])
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    st.pyplot(fig)

    st.subheader("🧠 Feature Influence Snapshot")
    st.write("**Form Score (home):**", round(form_score_home, 2))
    st.write("**Form Score (away):**", round(form_score_away, 2))
    st.write("**Goal Score (home):**", round(goals_score_home, 2))
    st.write("**Goal Score (away):**", round(goals_score_away, 2))
    st.write("**Net Star Power (home):**", round(star_power_home - missing_penalty_home, 2))
    st.write("**Net Star Power (away):**", round(star_power_away - missing_penalty_away, 2))
    st.write("**H2H Advantage (0-1):**", round(h2h_score, 2))
    st.write("**Intuition Boost:**", intuition_boost)
    st.write("**Match Importance:**", importance)
    st.write("**Possession Diff:**", home_possession - away_possession)
    st.write("**Shot Accuracy Diff:**", home_shot_accuracy - away_shot_accuracy)
    st.write("**xG / xGA (home):**", f"{home_xg}/{home_xga}")
    st.write("**xG / xGA (away):**", f"{away_xg}/{away_xga}")

    outcome_map = {"Home Win": f"{home_team} Win", "Draw": "Draw", "Away Win": f"{away_team} Win"}
    correctness = "✅" if (actual_result != "--" and result == outcome_map.get(actual_result)) else "❌" if actual_result != "--" else "N/A"

    st.session_state.history.append({
        "Match": f"{home_team} vs {away_team}",
        "Prediction": result,
        "Confidence": f"{confidence:.1f}%",
        "Actual": actual_result if actual_result != "--" else "(Not entered)",
        "Correct": correctness
    })

    if st.session_state.history:
        st.subheader("📊 Prediction History")
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)
        correct_preds = df[df['Correct'] == "✅"].shape[0]
        total_preds = df[df['Correct'].isin(["✅", "❌"])].shape[0]
        if total_preds > 0:
            st.success(f"Accuracy so far: {correct_preds}/{total_preds} ({(correct_preds/total_preds)*100:.1f}%)")
