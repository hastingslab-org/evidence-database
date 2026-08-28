DROP TABLE IF EXISTS qa_data;

CREATE TABLE qa_data (
    id VARCHAR(255) PRIMARY KEY,
    query TEXT,
    patient_data TEXT,
    papers TEXT,
    response TEXT,
    filters TEXT
);

