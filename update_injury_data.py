import json
from datetime import datetime
from nba_data_collector import NBADataCollector
import logging

def update_injury_data():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize collector (we don't need historical data for this)
    collector = NBADataCollector()
    
    try:
        # Just scrape current injuries
        logger.info("Collecting current injury data...")
        current_injuries = collector._scrape_injury_data()
        
        if current_injuries:
            injury_data = {
                'current': current_injuries,
                'metadata': {
                    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            }
            
            # Save to file
            output_path = f'{collector.output_dir}injury_data.json'
            with open(output_path, 'w') as f:
                json.dump(injury_data, f)
                
            logger.info(f"Successfully collected {len(current_injuries)} current injuries")
            return injury_data
            
        else:
            logger.error("No injury data collected")
            return None
            
    except Exception as e:
        logger.error(f"Error in update process: {str(e)}")
        return None

if __name__ == "__main__":
    print("Starting injury data update...")
    injury_data = update_injury_data()
    
    if injury_data:
        print("\nInjury data update complete!")
        print(f"Current injuries: {len(injury_data['current'])}")
        print(f"Last updated: {injury_data['metadata']['last_updated']}")
    else:
        print("\nFailed to update injury data")
