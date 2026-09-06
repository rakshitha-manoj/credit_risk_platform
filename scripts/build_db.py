import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data.loader import build_full_dataset
from src.ml.predict import predict
from src.talk_to_data.query_runner import build_db_from_dataframe
from src.utils.config import config
from src.utils.logger import get_logger
logger = get_logger(__name__)

def main():
    print('Step 1/3: Loading full dataset...')
    df = build_full_dataset(config.DATA_DIR)
    print(f'  Loaded {len(df)} applicants with {len(df.columns)} columns')
    print('Step 2/3: Scoring all applicants with trained model...')
    predictions = predict(df)
    print(f'  Scored {len(predictions)} applicants')
    print(f'  Risk band distribution:\n{predictions['risk_band'].value_counts().to_string()}')
    print('Step 3/3: Merging predictions and writing to SQLite...')
    merged = df.merge(predictions, on=config.ID_COL, how='left')
    build_db_from_dataframe(merged, config.SQL_DB_PATH)
    print(f'  Written {len(merged)} rows to {config.SQL_DB_PATH}')
    print('\nDone. The Talk-to-Data chatbot can now query real data.')
if __name__ == '__main__':
    main()