import streamlit as st

# --- Title ---
st.title("⚽ Soccer Match Predictor")

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

# --- Prediction ---
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
                    home_index = parts.index(home_team.split()[0])
                    away_index = parts.index(away_team.split()[0])

                    scores = [int(s.split('-')[0]) if '-' in s else None for s in parts if '-' in s or s.isdigit()]
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

    # --- Calculate Scores ---
    form_score_home = parse_form(home_form)
    form_score_away = parse_form(away_form)
    goals_score_home = home_goals_scored - home_goals_conceded
    goals_score_away = away_goals_scored - away_goals_conceded
    missing_penalty_home = parse_players(home_missing)
    missing_penalty_away = parse_players(away_missing)
    star_power_home = parse_players(home_stars)
    star_power_away = parse_players(away_stars)
    h2h_score = parse_h2h(h2h_results)

    # --- Final Team Scores ---
    score_home = (
        form_score_home * 0.2 +
        goals_score_home * 0.2 +
        (star_power_home - missing_penalty_home) * 0.2 +
        h2h_score * 0.15 +
        0.1 +  # home advantage
        0.15 * (star_power_home / (star_power_home + star_power_away + 0.01))
    )
    score_away = (
        form_score_away * 0.2 +
        goals_score_away * 0.2 +
        (star_power_away - missing_penalty_away) * 0.2 +
        (1 - h2h_score) * 0.15 +
        0.15 * (star_power_away / (star_power_home + star_power_away + 0.01))
    )

    confidence = abs(score_home - score_away) / (score_home + score_away + 0.01)
    result = "Draw"
    if score_home > score_away:
        result = f"{home_team} Win"
    elif score_away > score_home:
        result = f"{away_team} Win"

    # --- Display Result ---
    st.subheader("🏁 Prediction Result")
    st.write(f"**Predicted Result:** {result}")
    st.write(f"**Confidence Level:** {round(confidence * 100, 1)}%")

    # Optional score prediction (rough estimate based on goal stats)
    predicted_home_goals = round(home_goals_scored * (1 - missing_penalty_home / 20), 1)
    predicted_away_goals = round(away_goals_scored * (1 - missing_penalty_away / 20), 1)
    st.write(f"**Predicted Scoreline:** {home_team} {predicted_home_goals} - {predicted_away_goals} {away_team}")
