import os
import pandas as pd
import json

if __name__ == "__main__":
    data_path = "../../data/raw/register/data"

    dir = os.listdir(data_path)
    dir = [i for i in dir if "people" in i]

    blocks = []
    for name in dir:
        blocks.append(pd.read_csv(os.path.join(data_path, name), low_memory=False))
    players = pd.concat(blocks)

    players = players[["key_mlbam", "key_retro"]]
    players = players.dropna()
    player_mapping = dict(zip(players["key_mlbam"].astype(int), players["key_retro"]))

    # Save the dictionary to a JSON file
    json_file_path = "../../data/intermediate/mlb_to_retro_PlayerID_map.json"
    with open(json_file_path, "w") as f:
        json.dump(player_mapping, f)

    mlb_team_mapping = {
        110: "BAL",
        111: "BOS",
        147: "NYY",
        139: "TBR",
        141: "TOR",
        145: "CWS",
        114: "CLE",
        116: "DET",
        118: "KCR",
        142: "MIN",
        117: "HOU",
        108: "LAA",
        133: "OAK",
        136: "SEA",
        140: "TEX",
        144: "ATL",
        146: "MIA",
        121: "NYM",
        143: "PHI",
        120: "WSN",
        112: "CHC",
        113: "CIN",
        158: "MIL",
        134: "PIT",
        138: "STL",
        109: "ARI",
        115: "COL",
        119: "LAD",
        135: "SD",
        137: "SF",
    }

    json_file_path = "../../data/intermediate/mlb_to_retro_TeamID_map.json"
    with open(json_file_path, "w") as f:
        json.dump(mlb_team_mapping, f)
