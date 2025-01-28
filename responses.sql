DROP TABLE IF EXISTS qa_data;

CREATE TABLE qa_data (
    id TEXT PRIMARY KEY,
    query TEXT,
    patient_data TEXT,
    papers TEXT,
    response TEXT
);

