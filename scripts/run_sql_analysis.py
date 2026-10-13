
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[1]
SQL_FILE = ROOT / "sql" / "analysis_queries.sql"

sql_text = SQL_FILE.read_text()

# Split the file into individual statements.
queries = [
    query.strip()
    for query in sql_text.split(";")
    if query.strip()
]

con = duckdb.connect()

for i, query in enumerate(queries, start=1):
    print(f"\n{'=' * 65}")
    print(f"QUERY {i}")
    print("=" * 65)

    try:
        result = con.execute(query).df()
        print(result.to_string(index=False))
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise

con.close()

