import pandas as pd
import os
from nba_api.stats.endpoints import commonplayerinfo, leaguedashplayerstats
import time
from datetime import datetime

def generate_player_positions():
    """Generate a CSV file with player positions for recent seasons"""
    print("Fetching player data...")
    
    try:
        # Create data/raw directory if it doesn't exist
        os.makedirs('data/raw', exist_ok=True)
        
        # Test with one player first to see available columns
        print("Testing API with one player...")
        test_player = commonplayerinfo.CommonPlayerInfo(player_id=2544).get_data_frames()[0]  # LeBron James
        print("\nAvailable columns in player info:")
        print(test_player.columns.tolist())
        print("\nSample player data:")
        print(test_player.iloc[0])
        
        proceed = input("\nDo you want to proceed with gathering all player positions? (y/n): ")
        if proceed.lower() != 'y':
            print("Exiting...")
            return
            
        # If we proceed, continue with the full process
        print("\nGetting current season players...")
        current_players = leaguedashplayerstats.LeagueDashPlayerStats(
            season="2023-24"
        ).get_data_frames()[0]
        
        # Load your existing game logs to match players
        print("\nLoading game logs...")
        seasons = ['2022-23', '2023-24', '2024-25']
        game_log_players = set()
        
        for season in seasons:
            game_logs_path = f'data/historical/game_logs_{season}.csv'
            if os.path.exists(game_logs_path):
                game_logs = pd.read_csv(game_logs_path)
                if 'Player_ID' in game_logs.columns:
                    game_log_players.update(game_logs['Player_ID'].unique())
                elif 'PLAYER_ID' in game_logs.columns:
                    game_log_players.update(game_logs['PLAYER_ID'].unique())
        
        print(f"Found {len(game_log_players)} players in game logs")
        
        # Get position data for each player
        positions_data = []
        total_players = len(game_log_players)
        
        for idx, player_id in enumerate(game_log_players, 1):
            try:
                print(f"Processing player {idx}/{total_players}")
                player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
                
                if not player_info.empty:
                    positions_data.append({
                        'PLAYER_ID': player_id,
                        'PLAYER_NAME': player_info['DISPLAY_FIRST_LAST'].iloc[0],
                        'RAW_POSITION': player_info['POSITION'].iloc[0]
                    })
                
                # Respect API rate limits
                time.sleep(0.6)
                
            except Exception as e:
                print(f"Error fetching info for player {player_id}: {str(e)}")
                continue
        
        positions_df = pd.DataFrame(positions_data)
        
        # Clean up positions with more specific mapping
        position_mapping = {
            'Guard': 'G',
            'Guard-Forward': 'G',
            'Forward-Guard': 'G',
            'Forward': 'F',
            'Forward-Center': 'F',
            'Center-Forward': 'C',
            'Center': 'C',
            '': 'F'  # Default for empty positions
        }
        
        def clean_position(pos):
            if pd.isna(pos):
                return 'F'
            pos = str(pos).strip()
            # First try exact match
            if pos in position_mapping:
                return position_mapping[pos]
            # Then try partial match
            pos_upper = pos.upper()
            if 'GUARD' in pos_upper:
                return 'G'
            elif 'CENTER' in pos_upper:
                return 'C'
            return 'F'
        
        positions_df['POSITION'] = positions_df['RAW_POSITION'].apply(clean_position)
        final_df = positions_df[['PLAYER_ID', 'PLAYER_NAME', 'POSITION']]
        
        # Save to CSV
        output_path = 'data/raw/player_positions.csv'
        final_df.to_csv(output_path, index=False)
        print(f"\nSuccessfully saved player positions to {output_path}")
        print(f"Total players: {len(final_df)}")
        print("\nPosition distribution:")
        print(final_df['POSITION'].value_counts())
        
        # Print a sample of the data with raw positions
        print("\nSample of generated data (including raw positions):")
        print(positions_df[['PLAYER_ID', 'PLAYER_NAME', 'RAW_POSITION', 'POSITION']].head(10))
        
    except Exception as e:
        print(f"Error generating player positions: {str(e)}")
        raise

if __name__ == "__main__":
    generate_player_positions() 