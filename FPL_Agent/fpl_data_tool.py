from langchain_core.tools import tool
import pandas as pd
import requests

class FPLDataTool:
    """Tool to fetch Fantasy Premier League data."""
       
    def __init__(self):
        self.base_url = 'https://fantasy.premierleague.com/api/'
        self.columns = [
            'first_name', 'second_name', 'web_name', 'team_name', 'position_name',
            'now_cost', 'total_points', 'points_per_game', 'selected_by_percent',
            'minutes', 'expected_goals', 'expected_assists', 'expected_goal_involvements',
            'chance_of_playing_this_round', 'chance_of_playing_next_round', 'influence',
            'creativity', 'threat', 'ict_index'
        ]

    def get_player_data(self, player_name: str):
        """Fetches data for a player from the Fantasy Premier League API."""
        try:
            r = requests.get(self.base_url + 'bootstrap-static/').json()
        except Exception as e:
            return f"Error: {e}. Please try again."

        players = pd.DataFrame(r['elements'])
        teams = pd.DataFrame(r["teams"])
        positions = pd.DataFrame(r["element_types"])
        
        df = pd.merge(players, teams, left_on="team", right_on="id")
        df = pd.merge(df, positions, left_on="element_type", right_on="id")
        df = df.rename(columns={'name': 'team_name', 'singular_name': 'position_name'})
        
        names = player_name.split()
        if len(names) > 1:
            first_name, last_name = names
            player_stats = df[(df["first_name"].str.contains(first_name, case=False)) & 
                              (df["second_name"].str.contains(last_name, case=False))]
        else:
            player_stats = df[df["web_name"].str.contains(player_name, case=False)]

        if not player_stats.empty:
            return {key: list(value.values())[0] for key, value in player_stats[self.columns].to_dict().items()}
        else:
            return f"Player {player_name} not found."

fpl_tool = FPLDataTool()

@tool
def get_draft_players_data(player_names: list) :
    """Wraps FPLDataTool to get data for multiple players."""
    data = [fpl_tool.get_player_data(player_name) for player_name in player_names]
    return data
