import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# --- Title ---
st.title("⚽ Soccer Match Predictor (Paste Stats + ML + Intuition)")

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

# --- 📋 Paste-Based Stats Parser ---
st.subheader("📋 Paste Match Stats Block")
stats_block = st.text_area("Example:\nMatch: Man City vs Arsenal\nxG: 2.1 - 1.4\nGoals: 3 - 1\nPossession: 62 - 38\nShot Accuracy: 70 - 55", height=120)

auto_data = {}
if stats_block:
    try:
        lines = stats_block.strip().split('\n')
        for line in lines:
            if "match:" in line.lower():
                teams = line.split(":")[1].strip()
                home, away = [x.strip() for x in teams.split("vs")]
                auto_data["home_team"] = home
                auto_data["away_team"] = away
            elif "xg:" in line.lower():
                xg = [float(x.strip()) for x in line.split(":")[1].split("-")]
                auto_data["home_xg"], auto_data["away_xg"] = xg
            elif "goals:" in line.lower():
                goals = [float(x.strip()) for x in line.split(":")[1].split("-")]
                auto_data["home_goals"], auto_data["away_goals"] = goals
            elif "possession:" in line.lower():
                pos = [int(x.strip()) for x in line.split(":")[1].split("-")]
                auto_data["home_possession"], auto_data["away_possession"] = pos
            elif "shot accuracy:" in line.lower():
                acc = [int(x.strip()) for x in line.split(":")[1].split("-")]
                auto_data["home_shot_accuracy"], auto_data["away_shot_accuracy"] = acc
        st.success("✅ Stats block parsed successfully.")
    except:
        st.error("❌ Failed to parse. Check format.")

# --- Inputs ---
st.header("Team Info")
home_team = st.text_input("Home Team", auto_data.get("home_team", ""))
away_team = st.text_input("Away Team", auto_data.get("away_team", ""))

st.header("Team Stats")
home_goals_scored = st.number_input("Home Avg Goals Scored", value=auto_data.get("home_goals", 0.0))
home_goals_conceded = st.number_input("Home Avg Goals Conceded", value=0.0)
away_goals_scored = st.number_input("Away Avg Goals Scored", value=auto_data.get("away_goals", 0.0))
away_goals_conceded = st.number_input("Away Avg Goals Conceded", value=0.0)

home_possession = st.slider("Home Possession (%)", 0, 100, auto_data.get("home_possession", 50))
away_possession = st.slider("Away Possession (%)", 0, 100, auto_data.get("away_possession", 50))
home_shot_accuracy = st.slider("Home Shot Accuracy (%)", 0, 100, auto_data.get("home_shot_accuracy", 50))
away_shot_accuracy = st.slider("Away Shot Accuracy (%)", 0, 100, auto_data.get("away_shot_accuracy", 50))

home_xg = st.number_input("Home xG", value=auto_data.get("home_xg", 0.0))
home_xga = st.number_input("Home xGA", value=0.0)
away_xg = st.number_input("Away xG", value=auto_data.get("away_xg", 0.0))
away_xga = st.number_input("Away xGA", value=0.0)

st.header("Recent Form")
home_form = st.text_input("Home Form (e.g. W-W-L-D-W)")
away_form = st.text_input("Away Form")

st.header("Star / Missing Players")
home_stars = st.text_area("Home Star Players (name:rating)")
away_stars = st.text_area("Away Star Players (name:rating)")
home_missing = st.text_area("Home Missing Players (name:importance)")
away_missing = st.text_area("Away Missing Players (name:importance)")

st.header("Head-to-Head")
h2h_results = st.text_area("Last 5 matches (e.g. Arsenal 2-1 Liverpool)")

importance = st.selectbox("Match Importance", ["Normal", "Rivalry", "Cup Final"])
importance_score = {"Normal": 0, "Rivalry": 1, "Cup Final": 2}[importance]

st.header("Intuition Boost")
intuition_boost = st.slider("Favor home (-1.0) to away (+1.0)", -1.0, 1.0, 0.0, 0.1)

actual_result = st.selectbox("Actual Result (if known)", ["--", "Home Win", "Draw", "Away Win"])

# --- Prediction ---
if st.button("Predict Match Result"):
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
                    scores = [int(s) for part in line.split() for s in part.split('-') if s.isdigit()]
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

    form_home = parse_form(home_form)
    form_away = parse_form(away_form)
    goal_diff_home = home_goals_scored - home_goals_conceded
    goal_diff_away = away_goals_scored - away_goals_conceded
    star_home = parse_players(home_stars) - parse_players(home_missing)
    star_away = parse_players(away_stars) - parse_players(away_missing)
    h2h_score = parse_h2h(h2h_results)

    features = np.array([[form_home, form_away,
                          goal_diff_home, goal_diff_away,
                          star_home, star_away,
                          h2h_score, intuition_boost,
                          (home_possession - away_possession) / 100,
                          (home_shot_accuracy - away_shot_accuracy) / 100,
                          importance_score]])

    pred = clf.predict(features)[0]
    prob = clf.predict_proba(features)[0]
    result_label = ["Draw", f"{home_team} Win", f"{away_team} Win"]
    original_result = result_label[pred]
    confidence = round(np.max(prob) * 100, 1)

    net_home = goal_diff_home + star_home / 10 + form_home
    net_away = goal_diff_away + star_away / 10 + form_away
    total = net_home + net_away + 0.01
    score_home = round(max(0, 2.5 * (net_home / total)), 1)
    score_away = round(max(0, 2.5 * (net_away / total)), 1)

    if abs(score_home - score_away) <= 0.4:
        result = "Draw"
    elif score_home > score_away:
        result = f"{home_team} Win"
    else:
        result = f"{away_team} Win"

    st.subheader("🏁 Prediction Result")
    st.write(f"**Final Result:** {result}")
    st.write(f"**Model Confidence:** {confidence:.1f}%")
    st.write(f"**Predicted Scoreline:** {home_team} {score_home} - {score_away} {away_team}")
    st.write(f"**Raw Model Output:** {original_result}")

    st.subheader("📊 Probability Chart")
    fig, ax = plt.subplots()
    ax.bar(["Draw", f"{home_team} Win", f"{away_team} Win"], prob, color=['gray', 'green', 'red'])
    ax.set_ylim(0, 1)
    st.pyplot(fig)

    correctness = "✅" if (actual_result != "--" and result == original_result) else "❌" if actual_result != "--" else "N/A"
    st.session_state.history.append({
        "Match": f"{home_team} vs {away_team}",
        "Prediction": result,
        "Confidence": f"{confidence:.1f}%",
        "Actual": actual_result if actual_result != "--" else "(Not entered)",
        "Correct": correctness
    })

    if st.session_state.history:
        st.subheader("📚 Prediction History")
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)
        correct_preds = df[df["Correct"] == "✅"].shape[0]
        total_preds = df[df["Correct"].isin(["✅", "❌"])].shape[0]
        if total_preds:
            st.success(f"Accuracy: {correct_preds}/{total_preds} ({correct_preds / total_preds * 100:.1f}%)")
