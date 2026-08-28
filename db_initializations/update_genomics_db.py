from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from genomics.genomics_data import GenomicsData

civicData = GenomicsData(url=config.CIVIC_API_URL)
civicData.fetch_all_data()
civicData.save_to_sqlite(db_path=str(config.SQLITE_DB_PATH))
