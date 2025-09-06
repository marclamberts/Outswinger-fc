import os
import json
import pandas as pd
import numpy as np
import joblib

# ===========================
# Paths
# ===========================
json_folder = "/Users/user/XG/WSL/"           # JSON files
xT_path = "/Users/user/XT_grid.csv"           # xT grid
team_mapping_file = "/Users/user/XG/WSL Matches.csv"  # team mapping
output_csv = "/Users/user/Documents/GitHub/Outswinger-fc/data/WSL_xDisruption.csv"
model_path = "/Users/user/disruption_model.pkl"

os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# ===========================
# Load model
# ===========================
model = joblib.load(model_path)
print("✅ Loaded disruption model")

# ===========================
# Load xT grid
# ===========================
xT = pd.read_csv(xT_path, header=None).to_numpy()
xT_rows, xT_cols = xT.shape
print(f"xT grid loaded: {xT_rows} rows × {xT_cols} cols")

# ===========================
# Load team mapping
# ===========================
team_mapping_df = pd.read_csv(team_mapping_file)
home_teams = team_mapping_df[["matchInfo/contestant/0/id", "matchInfo/contestant/0/name"]].copy()
home_teams.columns = ["contestantId", "Team"]
away_teams = team_mapping_df[["matchInfo/contestant/1/id", "matchInfo/contestant/1/name"]].copy()
away_teams.columns = ["contestantId", "Team"]
team_map = pd.concat([home_teams, away_teams]).drop_duplicates()
id_to_team = dict(zip(team_map["contestantId"], team_map["Team"]))

# ===========================
# Helper to convert numeric safely
# ===========================
def safe_float(val):
    try:
        return float(val)
    except:
        return np.nan

# ===========================
# Aggregate all JSON files
# ===========================
all_data = []

for f in os.listdir(json_folder):
    if f.endswith(".json"):
        file_path = os.path.join(json_folder, f)
        with open(file_path, "r", encoding="utf-8") as jf:
            data = json.load(jf)
        events = data.get("event", data.get("events", []))
        rows = []
        for e in events:
            row = {
                "id": e.get("id"),
                "typeId": e.get("typeId"),
                "contestantId": str(e.get("contestantId")),
                "playerId": str(e.get("playerId")),
                "playerName": e.get("playerName"),
                "x": safe_float(e.get("x")),
                "y": safe_float(e.get("y")),
                "endX": np.nan,
                "endY": np.nan
            }
            for q in e.get("qualifier", []):
                qid = q.get("qualifierId")
                val = q.get("value", 1)
                if qid == 140: row["endX"] = safe_float(val)
                if qid == 141: row["endY"] = safe_float(val)
            rows.append(row)
        all_data.extend(rows)

df = pd.DataFrame(all_data)

# Fill missing coordinates
df[["x","y","endX","endY"]] = df[["x","y","endX","endY"]].fillna(0)

# --- Recipient, Passer, Receiver ---
df["recipientId"] = np.nan
for i in range(len(df)-1):
    if df.loc[i,"typeId"] == 1:
        next_row = df.loc[i+1]
        if next_row["contestantId"] == df.loc[i,"contestantId"]:
            df.at[i,"recipientId"] = next_row["playerId"]
df["recipientId"] = df["recipientId"].fillna(-1)
df["passer"] = df["playerId"]
df["receiver"] = df["recipientId"]

# --- Numeric features ---
df["distance"] = np.sqrt((df["endX"]-df["x"])**2 + (df["endY"]-df["y"])**2)
df["angle"] = np.abs(np.arctan2(df["endY"]-df["y"], df["endX"]-df["x"]))
df["xT"] = 0
df["xPass"] = 1 - (0.02*df["distance"] + 0.1*df["angle"])
df["xPass"] = df["xPass"].clip(0,1)
df["Pressures"] = 0
df["PP90"] = 0

# --- Preserve for stats ---
df_typeId = df["typeId"]
df_playerName = df["playerName"]
df_team = df["contestantId"].map(id_to_team)

# --- Dummies for model ---
df = pd.get_dummies(df, columns=["passer","receiver"], drop_first=True)

# --- Keep model features ---
model_features = model.feature_names_in_
for col in model_features:
    if col not in df.columns:
        df[col]=0
df_model = df[model_features]

# --- Predict ---
df_model["disruption_probability"] = model.predict_proba(df_model)[:,1]

# --- Aggregate player defensive ability ---
df_model["typeId"] = df_typeId
df_model["playerName"] = df_playerName
df_model["Team"] = df_team

player_stats = df_model.groupby(["playerName","Team"]).agg(
    TotalActions=("typeId","count"),
    ActualDisruptions=("typeId", lambda x: x.astype(str).isin(["7","8"]).sum()),
    ExpectedDisruptions=("disruption_probability","sum")
).reset_index()

# Scale ExpectedDisruptions by total actions
player_stats["ExpectedDisruptions"] = player_stats["ExpectedDisruptions"] * player_stats["TotalActions"]

# --- Save CSV with only defensive ability ---
player_stats[["playerName","Team","ActualDisruptions","ExpectedDisruptions"]].to_csv(output_csv, index=False)
print(f"✅ Saved defensive ability CSV to {output_csv}")
