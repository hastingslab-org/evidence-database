DROP TABLE IF EXISTS molecular_profiles;
CREATE TABLE IF NOT EXISTS molecular_profiles (
    id TEXT PRIMARY KEY,
    name TEXT,
    description TEXT,
    variants, TEXT,
    disease TEXT,
    molecularProfileScore REAL,
    db_source TEXT
);

DROP TABLE IF EXISTS genes;
CREATE TABLE IF NOT EXISTS genes (
    id TEXT PRIMARY KEY,
    name TEXT,
    description TEXT,
    variants JSON, -- Store list of variant IDs as a JSON array
    molecular_profiles JSON, -- Store list of molecular profile IDs as a JSON array
    diseases JSON,  -- Store list of disease IDs as a JSON array
    db_source TEXT
);

DROP TABLE IF EXISTS diseases;
CREATE TABLE IF NOT EXISTS diseases (
    id TEXT PRIMARY KEY,
    name TEXT,
    description TEXT,
);