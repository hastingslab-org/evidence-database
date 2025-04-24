from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent    #TODO this is a temporary fix, use packages structure or setup.cfg
sys.path.insert(0, str(ROOT))    

from genomics.genomics_data import GenomicsData

civicData = GenomicsData(url="https://civicdb.org/api/graphql")
civicData.fetch_all_data()
civicData.save_to_sqlite(db_path="../database.db")