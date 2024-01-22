import statsapi
from alive_progress import alive_bar


def fetch_mlb_play_by_play(start_date, end_date, TeamIDmap, PlayerIDmap):
    # Get the list of games on the specified date
    schedule = statsapi.schedule(start_date=start_date, end_date=end_date, sportId=1)

    # Loop through each game to fetch the play-by-play data
    games = []
    num_games = len(schedule)
    with alive_bar(num_games) as bar:
        for game in schedule:
            cur_game = {}

            # cur_game["game_id"] = game["game_id"]
            cur_game["away_team"] = TeamIDmap[game["away_id"]]
            cur_game["home_team"] = TeamIDmap[game["home_id"]]

            cur_game["regular_season"] = True if game["game_type"] == "R" else False
            cur_game["is_done"] = True if game["status"] == "Final" else False
            cur_game["home_score"] = game["home_score"]
            cur_game["away_score"] = game["away_score"]
            cur_game["plays"] = []

            # assemble retrosheet id
            date = game["game_date"]
            game_num = str(game["game_num"] - 1)
            year = str(date[:4])
            month = str(date[5:7])
            day = str(date[8:10])
            cur_game["game_id"] = cur_game["home_team"] + year + month + day + game_num
            cur_game["date"] = f"{year}/{month}/{day}"
            game_id = game["game_id"]
            # get the batting orders and starting pitchers for the game
            game_info = statsapi.get("game", {"gamePk": 748549})
            cur_game["home_sp"] = PlayerIDmap[
                game_info["gameData"]["probablePitchers"]["home"]["id"]
            ]
            cur_game["away_sp"] = PlayerIDmap[
                game_info["gameData"]["probablePitchers"]["away"]["id"]
            ]

            boxscore_data = statsapi.get("game_boxscore", {"gamePk": game_id})
            cur_game["home_batting_order"] = [
                PlayerIDmap[i] for i in boxscore_data["teams"]["home"]["battingOrder"]
            ]
            cur_game["away_batting_order"] = [
                PlayerIDmap[i] for i in boxscore_data["teams"]["away"]["battingOrder"]
            ]

            # get the play by play data for the game
            play_by_play_data = statsapi.get("game_playByPlay", {"gamePk": game_id})
            for play in play_by_play_data["allPlays"]:
                parse_play(play, cur_game, PlayerIDmap)

            # Add the play-by-play data to the dictionary, using the game ID as the key
            games.append(cur_game)
            bar()
        return games


def parse_play(play, cur_game, PlayerIDmap):
    if play["result"]["type"] != "atBat":
        return
    outcome = simplify_outcome(play["result"]["event"])
    if outcome == "Caught Stealing 2B":
        return
    batter = PlayerIDmap[play["matchup"]["batter"]["id"]]
    pitcher = PlayerIDmap[play["matchup"]["pitcher"]["id"]]
    num_pitches = len(play["pitchIndex"])
    cur_game["plays"].append(
        (
            {
                "batter": batter,
                "pitcher": pitcher,
                "outcome": outcome,
                "num_pitches": num_pitches,
                "raw_outcome": None,
            }
        )
    )


def simplify_outcome(outcome):
    outcome_map = {
        "Walk": "walk",
        "Pop Out": "popout",
        "Strikeout": "strikeout",
        "Single": "single",
        "Flyout": "flyout",
        "Lineout": "lineout",
        "Home Run": "homerun",
        "Groundout": "groundout",
        "Grounded Into DP": "groundout",
        "Intent Walk": "walk",
        "Hit By Pitch": "walk",
        "Sac Fly": "flyout",
        "Forceout": "groundout",
        "Field Error": "single",
        "Double": "double",
        "Double Play": "groundout",
        "Triple": "triple",
        "Fielders Choice": "groundout",
        "Catcher Interference": "walk",
        "Fielders Choice Out": "groundout",
        "Fielders Choice": "groundout",
        "Caught Stealing 2B": "unknown",
        "Sac Bunt": "groundout",
        "Strikeout Double Play": "Strikeout",
        "Bunt Pop Out": "popout",
        "Bunt Groundout": "groundout",
        "Bunt Lineout": "lineout",
        "Pickoff 1B": "unknown",
        "Pickoff Caught Stealing 2B": "unkown",
        "Caught Stealing Home": "unknown",
        "Runner Out": "unknown",
        "Sac Fly Double Play": "flyout",
        "Wild Pitch": "walk",
        "Caught Stealing 3B": "unknown",
        "Pickoff 2B": "unknown",
    }
    if outcome not in outcome_map:
        return "unknown"
    return outcome_map[outcome]


import json
import os
from pprint import pprint

# Example usage
# playerIDpath = "./baseball-MCS/data/intermediate/mlb_to_retro_PlayerID_map.json"
playerIDpath = "../../data/intermediate/mlb_to_retro_PlayerID_map.json"
with open(playerIDpath) as js:
    playerIDmap = json.load(js)
playerIDmap = {int(k): v for k, v in playerIDmap.items()}

# TeamIDpath = "./baseball-MCS/data/intermediate/mlb_to_retro_TeamID_map.json"
TeamIDpath = "../../data/intermediate/mlb_to_retro_TeamID_map.json"
with open(TeamIDpath) as js:
    TeamIDmap = json.load(js)
TeamIDmap = {int(k): v for k, v in TeamIDmap.items()}

start_date = "2023-04-03"
end_date = "2023-04-03"
all_games_path = "../../data/intermediate/all_games.json"

with open(all_games_path, "r") as f:
    all_games = json.load(f)

# List of dictionaries [{'game_id': xxx, 'plays': []}]
mlb_games = fetch_mlb_play_by_play(start_date, end_date, TeamIDmap, playerIDmap)
print("fetched from mlbapi")
all_gameids = set([game["game_id"] for game in all_games])
mlb_gameid_dict = {game["game_id"]: game for game in mlb_games}

# Update regular season games
for game in all_games:
    game_id = game["game_id"]
    if game_id in mlb_gameid_dict:
        # Update the dictionary
        game.update(mlb_gameid_dict[game_id])
print("updated existing games")
# Append games that are in mlb_gameid_dict but not in all_games
for game_id, game_data in mlb_gameid_dict.items():
    if game_id not in all_gameids:
        all_games.append(game_data)
print("added new games")
with open(all_games_path, "w+") as f:
    json.dump(all_games, f)
