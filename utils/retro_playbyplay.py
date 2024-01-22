from pprint import pprint
from collections import Counter
import json
import pandas as pd
import os
import numpy as np


def parse_id(game, lst):
    game["game_id"] = lst[1]


def parse_info(game, lst):
    if lst[1] == "visteam":
        game["away_team"] = lst[2]
    elif lst[1] == "hometeam":
        game["home_team"] = lst[2]
    elif lst[1] == "date":
        game["date"] = lst[2]


def parse_start_sub(game, lst):
    lst[5], lst[3], lst[4] = int(lst[5]), int(lst[3]), int(lst[4])
    team_key = "away" if lst[3] == 0 else "home"

    if lst[0] == "start":
        if lst[4] != 0:
            game[f"{team_key}_batting_order"][lst[4] - 1] = lst[1]
        if lst[5] == 1:
            game[f"{team_key}_sp"] = lst[1]

    if lst[5] == 1:
        game[f"cur_pitcher_{team_key}"] = lst[1]


# Takes in path to EVX file and returns list of dicts. Each element in list is a game. Keys
# and values are parsed based on functions above. plays are stored as a list of each play.
def parse_EVX(path):
    all_games = []
    first = 1
    with open(path) as f:
        for line in f:
            line = line.strip("\n")
            line = line.split(",")

            match line[0]:
                case "id":
                    if not first:
                        del cur_game["cur_pitcher_away"]
                        del cur_game["cur_pitcher_home"]
                        all_games.append(cur_game)
                    first = 0
                    cur_game = {
                        "plays": [],
                        "home_batting_order": [None for i in range(9)],
                        "away_batting_order": [None for i in range(9)],
                        "is_done": True,
                        "regular_season": True,
                    }
                    parse_id(cur_game, line)
                case "info":
                    parse_info(cur_game, line)
                case "start":
                    parse_start_sub(cur_game, line)
                case "sub":
                    parse_start_sub(cur_game, line)
                case "play":
                    parse_play(cur_game, line)
    del cur_game["cur_pitcher_away"]
    del cur_game["cur_pitcher_home"]
    all_games.append(cur_game)
    return all_games


# Extended parse_outcome function
def parse_outcome(outcome_code):
    outcome_mapping = {
        "K": "strikeout",
        "S": "single",
        "D": "double",
        "T": "triple",
        "H": "homerun",
        "W": "walk",
        "GO": "groundout",
        "FO": "flyout",
        "HP": "walk",  # Hit by Pitch, categorized as a walk
        "/G": "groundout",
        "/F": "flyout",
        "/L": "lineout",
        "/P": "popout",
    }

    for code, full_word in outcome_mapping.items():
        if code in outcome_code:
            return full_word

    return "unknown"  # Default if the code is not recognized


# Updated parse_play function
def parse_play(game, lst):
    lst[2] = int(lst[2])
    lst[1] = int(lst[1])
    cur_pitcher = game["cur_pitcher_home"] if lst[2] == 1 else game["cur_pitcher_away"]
    num_pitches = -99 if lst[5] == "" else len(lst[5])
    outcome = parse_outcome(lst[6])
    play = {
        "inning": lst[1],
        "pitcher": cur_pitcher,
        "batter": lst[3],
        "num_pitches": num_pitches,
        "outcome": outcome,
        "raw_outcome": lst[6],
    }
    if "plays" not in game:
        game["plays"] = []
    game["plays"].append(play)


# Enhanced function to count the frequency of each outcome type across all games
# and provide additional statistics and sanity checks
def analyze_outcomes(games):
    outcome_counter = Counter()
    total_plays = 0
    total_games = len(games)

    # Counting outcomes and total plays
    for game in games:
        for play in game.get("plays", []):
            outcome_counter[play["outcome"]] += 1
            total_plays += 1

    # Calculating outcome probabilities
    outcome_probabilities = {
        outcome: count / total_plays for outcome, count in outcome_counter.items()
    }

    # Sanity Checks
    sanity_checks = {
        "prob_sum": round(sum(outcome_probabilities.values()), 2) == 1.0,
        "missing_home_team_or_date": any(
            "home_team" not in game or "date" not in game for game in games
        ),
        "games_with_no_plays": sum(1 for game in games if not game.get("plays")),
    }

    # Summary Stats
    summary_stats = {
        "total_games": total_games,
        "total_plays": total_plays,
    }

    # Combine all stats into one dictionary for easier output
    all_stats = {
        "Outcome Counts": outcome_counter,
        "Outcome Probabilities": outcome_probabilities,
        "Summary Stats": summary_stats,
        "Sanity Checks": sanity_checks,
    }

    pprint(all_stats)


def identify_unknown_codes(games):
    unknown_codes = Counter()

    for game in games:
        for play in game.get("plays", []):
            if play["outcome"] == "unknown":
                # Assuming the original outcome code is stored in a 'raw_outcome' field in the play dictionary
                # If it's stored differently, this line should be adjusted
                unknown_codes[
                    play.get("raw_outcome", "N/A")
                ] += 1  # 'N/A' indicates that the raw outcome is missing

    return unknown_codes


def aggregate_EVX_files(root_directory):
    all_games = []
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file.endswith(".EVA") or file.endswith(".EVN"):
                file_path = os.path.join(root, file)
                games = parse_EVX(file_path)
                all_games.extend(games)
    return all_games


def add_scores_to_games(all_games, gamelogs_dir):
    blocks = []
    for root, dirs, files in os.walk(gamelogs_dir):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                blocks.append(pd.read_csv(file_path, header=None))
    game_logs = pd.concat(blocks)

    # switch columns where home team bats first
    game_logs["game_id"] = np.where(
        game_logs.iloc[:, 159] == "HTBF",  # Condition
        game_logs[3].astype(str)
        + game_logs[0].astype(str)
        + game_logs[1].astype(str),  # True case
        game_logs[6].astype(str)
        + game_logs[0].astype(str)
        + game_logs[1].astype(str),  # False case
    )
    # map {game_id : [home_score, away_score]}
    scores = zip(game_logs[10].astype(int), game_logs[9].astype(int))
    run_mapping = dict(zip(game_logs["game_id"], scores))
    all_games = [game for game in all_games if game["game_id"] in run_mapping]
    for game in all_games:
        game["home_score"] = run_mapping[game["game_id"]][0]
        game["away_score"] = run_mapping[game["game_id"]][1]

    return all_games


def EVX_json_to_csv(path):
    with open(path, "r") as f:
        all_games = json.load(f)

    rows = []
    for game in all_games:
        for play in game.get("plays", []):
            rows.append(
                {
                    "pitcher": play["pitcher"],
                    "batter": play["batter"],
                    "outcome": play["outcome"],
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv("plays.csv")


if __name__ == "__main__":
    playbyplay_directory = "/home/projects/baseball-MCS/data/raw/playbyplay"  # Replace with your actual root directory
    gamelogs_directory = "/home/projects/baseball-MCS/data/raw/gamelogs"
    output_path = "/home/projects/baseball-MCS/data/intermediate/all_games.json"
    all_games = aggregate_EVX_files(playbyplay_directory)
    print(f"found {len(all_games)} games")
    all_games = add_scores_to_games(all_games, gamelogs_directory)
    print(f"scores for {len(all_games)}")
    # # analyze_outcomes(all_games)

    # # # Save the object as a JSON file
    with open(output_path, "w+") as f:
        json.dump(all_games, f)
    with open(output_path, "r+") as f:
        test_games = json.load(f)
