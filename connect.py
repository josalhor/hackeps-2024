import geopandas as gpd
from sqlalchemy import create_engine

# Database connection parameters
DB_USER = "user"
DB_PASSWORD = "password"
DB_NAME = "database"
DB_HOST = "localhost"  # Change to your DB host if different
DB_PORT = "5432"       # Default PostgreSQL port

# Connection string
connection_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(connection_string)

def load_table(table_name):
    """Load a spatial table into a GeoDataFrame."""
    query = f"SELECT * FROM {table_name}"
    try:
        gdf = gpd.read_postgis(query, con=engine, geom_col='geom')
        print(f"Loaded {table_name} with {len(gdf)} records.")
        return gdf
    except Exception as e:
        print(f"Failed to load {table_name}: {e}")
        return None

# Table names
tables = {
    "final": "eps.final",
    "inicio": "eps.inicio",
    "red1": "eps.red1",
    "red2": "eps.red2",
    "red3": "eps.red3",
    "red1_puntos": "eps.red1_puntos",
    "red2_puntos": "eps.red2_puntos",
    "red3_puntos": "eps.red3_puntos"
}

# Load all tables into memory
data = {name: load_table(tables[name]) for name in tables}

# Example: Accessing loaded tables
final = data["final"]
inicio = data["inicio"]
red1 = data["red1"]

# Print loaded table info
for table, gdf in data.items():
    if gdf is not None:
        print(f"{table}: {len(gdf)} records loaded.")
    else:
        print(f"{table}: Failed to load.")
