DROP TABLE IF EXISTS molecular_profiles;
CREATE TABLE IF NOT EXISTS molecular_profiles (
    id TEXT PRIMARY KEY,
    name TEXT,
    description TEXT,
    disease TEXT,
    molecularProfileScore REAL
)