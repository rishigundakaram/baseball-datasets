"""
Load all the schedule files and all_games.json. 
If all_games doesn't have a certain game, add it then change the file 
"""
import os
import pandas as pd
import json

if __name__ == "__main__":
    schedule_path = "../../data/raw/schedules"
    all_games_path = "../../data/intermediate/all_games.json"

    dir = os.listdir(schedule_path)
    dir = [i for i in dir if "schedule" in i]

    blocks = []
    for name in dir:
        blocks.append(
            pd.read_csv(
                os.path.join(schedule_path, name), low_memory=False, header=None
            )
        )
    schedule = pd.concat(blocks)
    schedule["game_id"] = (
        schedule[6].astype(str) + schedule[0].astype(str) + schedule[1].astype(str)
    )
    scheduled_games = dict(
        zip(schedule["game_id"], zip(schedule[0].astype(str), schedule[6], schedule[3]))
    )
    with open(all_games_path, "r") as f:
        all_games = json.load(f)
    all_gameids = set([game["game_id"] for game in all_games])
    for gameid, values in scheduled_games.items():
        if gameid not in all_gameids:
            all_games.append(
                {
                    "is_done": False,
                    "date": f"{values[0][:4]}/{values[0][4:6]}/{values[0][6:]}",
                    "game_id": gameid,
                    "home_team": values[1],
                    "away_team": values[2],
                }
            )
    with open(all_games_path, "w+") as f:
        json.dump(all_games, f)
