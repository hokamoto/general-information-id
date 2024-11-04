-- Main table for General Information ID (GIID) entries
CREATE TABLE giid_entries (
    giid TEXT PRIMARY KEY, -- GIID as a unique identifier
    date_identified TEXT,  -- Date identified or published
    narrative TEXT         -- Full sample narrative text as a JSON array
);

-- Table for language distribution percentages for each GIID entry
CREATE TABLE language_distribution (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    giid TEXT,
    language TEXT,
    percentage INTEGER,
    FOREIGN KEY (giid) REFERENCES giid_entries(giid)
);

-- Table for tags associated with each GIID entry
CREATE TABLE tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    giid TEXT,
    tag TEXT,
    FOREIGN KEY (giid) REFERENCES giid_entries(giid)
);

-- Table for evaluations by various organizations for each GIID entry
CREATE TABLE evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    giid TEXT,
    organization TEXT,
    evaluation TEXT,       -- Evaluation (True, False, Misleading, etc.)
    date_of_evaluation TEXT,
    comments TEXT,         -- Comments provided by the organization
    link TEXT,             -- Link to the source
    FOREIGN KEY (giid) REFERENCES giid_entries(giid)
);


create virtual table vec_giid using vec0(
  giid text primary key,
  embedding float[384]
);
