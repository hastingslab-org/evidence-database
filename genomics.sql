CREATE TABLE IF NOT EXISTS molecular_profiles (
    id INTEGER PRIMARY KEY,
    name TEXT,
    description TEXT,
    variants, TEXT,
    disease TEXT,
    molecularProfileScore REAL,
    db_source TEXT
);

CREATE TABLE IF NOT EXISTS genes (
    id INTEGER PRIMARY KEY,
    name TEXT,
    description TEXT,
    variants INTEGER[], -- Store list of variant IDs as a JSON array
    molecular_profiles INTEGER[], -- Store list of molecular profile IDs as a JSON array
    diseases INTEGER[],  -- Store list of disease IDs as a JSON array
    db_source TEXT
);

CREATE TABLE IF NOT EXISTS diseases (
    id INTEGER PRIMARY KEY,
    name TEXT,
    db_source TEXT
);


CREATE TABLE IF NOT EXISTS variants (
    id INTEGER PRIMARY KEY,
    name TEXT,
    description TEXT,
    gene_id TEXT,
    molecular_profiles INTEGER[], -- Store list of molecular profile IDs as a JSON array
    diseases INTEGER[], 
    db_source TEXT
);
