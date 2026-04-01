# ASMI Storage – SQL Query Examples

After running measurements or backfilling, data is stored in:

| Table | Purpose |
|-------|---------|
| `asmi_runs` | One row per run folder (run_XXX_YYYYMMDD_HHMMSS) |
| `asmi_measurement_files` | One row per well CSV; includes `csv_content` (full CSV text) |
| `asmi_plot_files` | One row per plot file (PNG, summary.csv, summary.txt); includes `file_path`, optional `file_content` |
| `asmi_well_summary` | Parsed summary per well (ElasticModulus, Std, R2) for easy querying |

---

## Setup

```bash
cd ASMI_new
python -m sql_interface.create_tables
```

Backfill existing runs:

```bash
python scripts/backfill_runs_to_db.py
# Or specific runs:
python scripts/backfill_runs_to_db.py run_732_20251030_122001 run_738_20260228_083050
```

---

## Example SQL Queries

### List all runs

```sql
SELECT run_id, run_folder_name, run_count, created_at, substrate_id, asmi_job_id
FROM asmi_runs
ORDER BY created_at DESC;
```

### Get elastic modulus by well across runs

```sql
SELECT r.run_folder_name, s.well_id, s.elastic_modulus, s.elastic_modulus_original, s.std, s.r2
FROM asmi_well_summary s
JOIN asmi_runs r ON r.run_id = s.run_id
ORDER BY r.created_at DESC, s.well_id;
```

### Get elastic modulus for a specific well (e.g. E5)

```sql
SELECT r.run_folder_name, r.created_at, s.elastic_modulus, s.std, s.r2
FROM asmi_well_summary s
JOIN asmi_runs r ON r.run_id = s.run_id
WHERE s.well_id = 'E5'
ORDER BY r.created_at DESC;
```

### Get full CSV content for a measurement

```sql
SELECT m.well_id, m.file_path, m.csv_content, m.elastic_modulus, m.contact_z
FROM asmi_measurement_files m
JOIN asmi_runs r ON r.run_id = m.run_id
WHERE r.run_folder_name = 'run_738_20260228_083050' AND m.well_id = 'A1';
```

### List all plot files for a run

```sql
SELECT p.well_id, p.plot_type, p.file_path
FROM asmi_plot_files p
JOIN asmi_runs r ON r.run_id = p.run_id
WHERE r.run_folder_name = 'run_732_20251030_122001'
ORDER BY p.well_id, p.plot_type;
```

### Join with asmi task table (when from task loop)

```sql
SELECT a.job_id, a.substrate_id, a.well_id, a.status, a.force, a.`z-position`,
       r.run_folder_name, s.elastic_modulus, s.std
FROM asmi a
LEFT JOIN asmi_runs r ON r.asmi_job_id = a.job_id
LEFT JOIN asmi_well_summary s ON s.run_id = r.run_id AND s.well_id = a.well_id
WHERE a.time_completed IS NOT NULL
ORDER BY a.time_completed DESC;
```

### Average elastic modulus per well across runs

```sql
SELECT well_id,
       AVG(elastic_modulus) AS avg_E,
       STDDEV(elastic_modulus) AS std_E,
       COUNT(*) AS n_runs
FROM asmi_well_summary
GROUP BY well_id;
```
