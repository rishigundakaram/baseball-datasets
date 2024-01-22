import json
from pprint import pprint
import unittest
from retro_playbyplay import parse_EVX, add_scores_to_games

all_games_path = "/home/projects/baseball-MCS/data/intermediate/all_games.json"

with open(all_games_path, "r") as f:
    all_games = json.load(f)


def duplicates(all_games):
    game_keys = set()
    for game in all_games:
        if game["game_id"] not in game_keys:
            game_keys.add(game["game_id"])
        else:
            pprint(all_games)
            raise Exception
    print("no duplicates")


def import_quality(all_games):
    game_keys = [
        "home_team",
        "away_team",
        "date",
        "home_batting_order",
        "away_batting_order",
        "home_sp",
        "away_sp",
        "home_score",
        "away_score",
        "regular_season",
    ]
    for game in all_games:
        for key in game_keys:
            if key not in game.keys():
                pprint(game)
                raise Exception
    print("passed import quality")


class TestParseEVX(unittest.TestCase):
    def test_num_games(self):
        all_games_path = "/home/projects/baseball-MCS/data/test/test.EVA"
        all_games = parse_EVX(all_games_path)
        self.assertEqual(len(all_games), 3)

    def test_2011(self):
        all_games_path = (
            "/home/projects/baseball-MCS/data/raw/playbyplay/2011eve/2011FLO.EVN"
        )
        all_games = parse_EVX(all_games_path)
        self.assertEqual(len(all_games), 81)

    def test_htbf(self):
        all_games_path = "/home/projects/baseball-MCS/data/test/htbf.EVA"
        gamelogs_dir = "/home/projects/baseball-MCS/data/test/gamelogs"
        all_games = parse_EVX(all_games_path)
        self.assertEqual(len(all_games), 1)
        all_games = add_scores_to_games(all_games, gamelogs_dir)
        self.assertEqual(len(all_games), 1)
        self.assertIn("home_score", all_games[0].keys())
        self.assertIn("away_score", all_games[0].keys())


class TestDataStructure(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # This is called once before all tests in this class
        all_games_path = "/home/projects/baseball-MCS/data/intermediate/all_games.json"
        with open(all_games_path, "r") as f:
            self.all_games = json.load(f)

        mlb_to_retro_TeamID_map_path = (
            "/home/projects/baseball-MCS/data/intermediate/mlb_to_retro_TeamID_map.json"
        )
        with open(mlb_to_retro_TeamID_map_path, "r") as f:
            self.team_map = json.load(f)

    def test_duplicates(self):
        game_keys = set()
        for game in all_games:
            if game["game_id"] not in game_keys:
                game_keys.add(game["game_id"])
            else:
                pprint(all_games)
                raise Exception

    def test_import_quality(self):
        game_keys = set(
            [
                "home_team",
                "away_team",
                "date",
                "home_batting_order",
                "away_batting_order",
                "home_sp",
                "away_sp",
                "home_score",
                "away_score",
                "regular_season",
                "plays",
                "is_done",
                "game_id",
            ]
        )
        for game in all_games:
            a = set(game.keys())
            assert (
                game_keys == a
            ), f"actual differences {game_keys - a}, {a - game_keys}, actual game {pprint(game)}"

    def test_team_names(self):
        team_names = set(self.team_map.values())
        for game in self.all_games:
            assert (
                game["home_team"] in team_names and game["away_team"] in team_names
            ), f'gameID: {game["game_id"]}, teams: {game["home_team"], game["away_team"]}'

    def test_year_game_counts(self):
        team_names = set(self.team_map.values())
        years = {i: {team: 0 for team in team_names} for i in range(2011, 2024)}

        for game in self.all_games:
            date = int(game["date"][:4])
            home_team = game["home_team"]
            away_team = game["away_team"]
            years[date][home_team] += 1
            years[date][away_team] += 1

        for year in years:
            for val in years[year].values():
                if val == 0 or year == 2020 or val >= 160:
                    continue
                pprint(years)
                assert False


if __name__ == "__main__":
    unittest.main()
