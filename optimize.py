import geopandas as gpd
import pandas as pd
import random
from shapely.geometry import Point, MultiLineString, Point
from sqlalchemy import create_engine
from geopy.distance import geodesic
from sqlalchemy import inspect
from geoalchemy2 import Geometry
from sqlalchemy.sql import text

# Database connection parameters
DB_USER = "user"
DB_PASSWORD = "password"
DB_NAME = "database"
DB_HOST = "localhost"
DB_PORT = "5432"

# Connection string
connection_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(connection_string)

# Load tables
def load_table(table_name):
    """Load a spatial table into a GeoDataFrame."""
    query = f"SELECT * FROM {table_name}"
    return gpd.read_postgis(query, con=engine, geom_col="geom")

# Load data
print('Loading tables...')
inicio = load_table("eps.inicio")
final = load_table("eps.final")
red1 = load_table("eps.red1")
red2 = load_table("eps.red2")
red3 = load_table("eps.red3")
red1_puntos = load_table("eps.red1_puntos")
red2_puntos = load_table("eps.red2_puntos")
red3_puntos = load_table("eps.red3_puntos")

# Inicio and final points are in a different format
# than the red points
# Inicio points are in EPSG:4326, while network points are in EPSG:25830
# We need to convert the inicio and final points to EPSG:25830

def convert_to_25830(gdf):
    """Convert a GeoDataFrame to EPSG:25830."""
    return gdf.to_crs(epsg=25830)

# Convert inicio and final points to EPSG:25830
inicio = convert_to_25830(inicio)
final = convert_to_25830(final)

# For some reason, red3_puntos is made of 
# MultiPoint geometries instead of Point geometries
# But the multi-point geometries are just single points
# So we can convert them to Point geometries

def convert_multipoint_to_point(mp):
    """Convert a MultiPoint to a Point."""
    return mp.geoms[0]
red3_puntos["geom"] = red3_puntos["geom"].apply(convert_multipoint_to_point)

def create_multilinestring_from_points(visited_points):
    """
    Generates a MULTILINESTRING WKT from a list of visited points.
    
    :param visited_points: List of Shapely Point objects representing the path.
    :return: WKT representation of the MULTILINESTRING.
    """
    # Create line segments between consecutive points
    line_segments = [(visited_points[i], visited_points[i + 1]) for i in range(len(visited_points) - 1)]
    
    # Create a MultiLineString object
    multilinestring = MultiLineString(line_segments)
    
    # Return WKT representation
    return multilinestring.wkt

# def ensure_results_table_exists(engine):
#     """
#     Ensures that the 'results' table exists in the database.
#     If it does not exist, creates the table with the required schema.
#     """
#     table_exists_query = text("""
#     SELECT EXISTS (
#         SELECT 1
#         FROM information_schema.tables
#         WHERE table_schema = 'eps' AND table_name = 'results'
#     );
#     """)
    
#     with engine.connect() as conn:
#         result = conn.execute(table_exists_query).scalar()
#         if not result:
#             create_table_query = text("""
#             CREATE TABLE eps.results (
#                 id SERIAL PRIMARY KEY,
#                 start_point GEOMETRY(Point, 25830) NOT NULL,
#                 end_point GEOMETRY(Point, 25830) NOT NULL,
#                 path GEOMETRY(MultiLineString, 25830) NOT NULL,
#                 distance FLOAT NOT NULL,
#                 UNIQUE (start_point, end_point)
#             );
#             """)
#             conn.execute(create_table_query)
#             print("Table 'eps.results' created.")
#             result = conn.execute(table_exists_query).scalar()
#             assert result, f"Table 'eps.results' was not created successfully."
#         else:
#             print("Table 'eps.results' already exists.")



def insert_or_update_result(engine, start_point_wkt, end_point_wkt, path_wkt, distance):
    """
    Inserts or updates a result into the 'results' table.
    If a row with the same start_point and end_point exists, updates the path and distance.
    """
    query_insert = text("""
    INSERT INTO eps.results (start_point, end_point, path, distance)
    VALUES (
        ST_GeomFromText(:start_point_wkt, 25830),
        ST_GeomFromText(:end_point_wkt, 25830),
        ST_GeomFromText(:path_wkt, 25830),
        :distance
    )
    ON CONFLICT (start_point, end_point)
    DO UPDATE SET path = EXCLUDED.path, distance = EXCLUDED.distance;
    """)
    
    with engine.connect() as conn:
        with conn.begin():  # Start a transaction
            conn.execute(query_insert, {
                "start_point_wkt": start_point_wkt,
                "end_point_wkt": end_point_wkt,
                "path_wkt": path_wkt,
                "distance": distance
            })
            print(f"Result for ({start_point_wkt}, {end_point_wkt}) inserted/updated.")

            query_id = text("""
            SELECT id FROM eps.results
            WHERE start_point = ST_GeomFromText(:start_point_wkt, 25830)
            AND end_point = ST_GeomFromText(:end_point_wkt, 25830);
            """)
            result = conn.execute(query_id, {
                "start_point_wkt": start_point_wkt,
                "end_point_wkt": end_point_wkt
            }).scalar()
            print(f"Result ID: {result}")


# Combine all red points into one GeoDataFrame
# all_points = gpd.GeoDataFrame(
#     pd.concat([red1_puntos, red2_puntos, red3_puntos], ignore_index=True),
#     crs=red1_puntos.crs,
#     geometry="geom"
# )
points_by_network = {
    "red1": red1_puntos,
    "red2": red2_puntos,
    "red3": red3_puntos
}

# Organize the networks into a dictionary
networks = {
    "red1": red1,
    "red2": red2,
    "red3": red3
}

# Select random start and end points
start_point = inicio.sample(1, random_state=42).iloc[0]
end_point = final.sample(1, random_state=42).iloc[0]

print(f"Start point ID: {start_point.id}, End point ID: {end_point.id}")

# Parameters
search_radius_network = 1000  # in meters
increase_radius_network = 1000  # in meters
search_radius_other_networks = 200  # in meters
max_search_radius = 5000  # in meters

# This works for points in format 4326 not 25830
# def haversine_distance(point1, point2):
#     """Calculate the haversine distance (meters) between two points."""
#     return geodesic((point1.y, point1.x), (point2.y, point2.x)).meters

# This works for points in format 25830
def haversine_distance(point1, point2):
    """Calculate the haversine distance (meters) between two points."""
    return point1.distance(point2)


def get_adjacent_points_and_lines(point, endpoint, points_by_network, red_lines, current_red, search_radius_network, increase_radius_network, search_radius_other_networks):
    """
    Find all candidates within the search radius, including:
    - Points from the same or other networks within the radius.
    - Adjacent points in the same network.
    """
    print(f"Finding adjacent points for {point}")
    # Find points in other networks within the radius
    # nearby_points = red_points[red_points.distance(point) <= radius]
    points_current_network = pd.DataFrame()
    heuristic_current_point = haversine_distance(point, endpoint)
    while points_current_network.empty and search_radius_network <= max_search_radius:
        points_current_network = points_by_network[current_red][points_by_network[current_red].distance(point) <= search_radius_network].copy()
        search_radius_network += increase_radius_network
        points_current_network = points_current_network[points_current_network['geom'].apply(lambda pt: haversine_distance(pt, endpoint) < heuristic_current_point)]
    points_current_network['network'] = current_red

    # Filter those with a heuristic less than the current point

    all_points = points_current_network

    # Find points in other networks within the radius
    while all_points.empty:
        for network_name, network in points_by_network.items():
            if network_name != current_red:
                nearby_points = network[network.distance(point) <= search_radius_other_networks].copy()
                nearby_points['network'] = network_name
                all_points = pd.concat([all_points, nearby_points])
        search_radius_other_networks += increase_radius_network
    
    return all_points
    # Find adjacent points along the same red
    # adjacent_points = []
    # for line in red_lines[current_red].itertuples():
    #     if line.geom.distance(point) <= radius:
    #         # Extract points (start and end) of the line segment
    #         coords = list(line.geom.coords)
    #         for i, coord in enumerate(coords):
    #             candidate = Point(coord)
    #             if point.distance(candidate) <= radius:
    #                 adjacent_points.append(candidate)
    #             # Add next or previous points in the sequence
    #             if i > 0:
    #                 adjacent_points.append(Point(coords[i - 1]))
    #             if i < len(coords) - 1:
    #                 adjacent_points.append(Point(coords[i + 1]))
    
    # Combine and return unique points
    # return gpd.GeoDataFrame(
    #     geometry=list(set(nearby_points["geom"].tolist() + adjacent_points))
    # )

def get_adjacent_points(point, red_points, radius):
    """ Find all candidates within the search radius. """
    print(f"Finding adjacent points for {point} within {radius} meters...")
    nearby_points = red_points[red_points.distance(point) <= radius]
    return nearby_points

def find_closest_to_network(point, points_by_network):
    """Find the closest point in the network to the given point."""
    candidates = pd.DataFrame()
    for network_name, network in points_by_network.items():
        new_candidates = network.copy()
        new_candidates["network"] = network_name
        candidates = pd.concat([candidates, new_candidates])
    
    candidates.reset_index(drop=True, inplace=True)
    # Find the closest point
    closest_point = candidates.loc[
        candidates.distance(point).idxmin()
    ]
    # print(f"Closest point found: {closest_point}")
    return closest_point.geom, closest_point.network

def best_first_search(start, end, points_by_network, networks, search_radius_network, increase_radius_network, search_radius_other_networks):
    """Perform Best-First Search."""
    visited = []
    total_distance = 0  # Track total distance
    current_red = None  # Track current red network
    print('Start algorithm')

    # For the first and last point, we need to find the closest point in the network
    # Since it is not guaranteed to be in the network
    # To do this, we will increment the search radius until we find a point
    print('Finding closest points in the network...')
    real_start, start_network = find_closest_to_network(start, points_by_network)
    real_end, end_network = find_closest_to_network(end, points_by_network)
    print('Closest start points found:', start, real_start, 'Distance:', haversine_distance(start, real_start))
    print('Closest end points found:', end, real_end, 'Distance:', haversine_distance(end, real_end))
    visited.append(start)
    current_point = real_start
    current_red = start_network
    total_distance += haversine_distance(start, real_start)
    
    print('Starting search on network:', current_red)
    while haversine_distance(current_point, real_end) > search_radius_network:
        visited.append(current_point)
        
        # Get all candidates (nearby points and adjacent points in the same network)
        candidates = get_adjacent_points_and_lines(
            current_point,
            real_end,
            points_by_network,
            networks,
            current_red,
            search_radius_network, increase_radius_network, search_radius_other_networks
        )
        
        if candidates.empty:
            print("No path found!")
            return visited, total_distance
        
        # Calculate heuristic for each candidate
        candidates["heuristic"] = candidates["geom"].apply(
            lambda pt: haversine_distance(pt, end)
        )
        
        # Choose the best candidate based on heuristic
        best_candidate = candidates.sort_values("heuristic").iloc[0]
        current_red = best_candidate.network
        best_candidate = best_candidate.geom
        
        # Update total distance
        total_distance += haversine_distance(current_point, best_candidate)
        current_point = best_candidate
    
    # Add the end point to the path
    visited.append(real_end)
    visited.append(end)
    total_distance += haversine_distance(current_point, end)
    
    print("Path found!")
    return visited, total_distance

def print_reslts_table(engine):
    query = text("""
    SELECT * FROM eps.results;
    """)
    with engine.connect() as conn:
        results = conn.execute(query).fetchall()
        # for result in results:
        #     print(result)
        print(f"Results table has {len(results)} records.")
        if len(results) > 0:
            print("Results:")
            for result in results:
                print(f"ID: {result.id}, Start: {result.start_point}, End: {result.end_point}, Distance: {result.distance:.2f} meters")
        else:
            print("No results found.")
    exit()

if __name__ == "__main__":
    # print_reslts_table(engine)
    print("Start point:", start_point)
    print("End point:", end_point)
    # ensure_results_table_exists(engine)
    # Run Best-First Search
    path, total_distance = best_first_search(
        start_point.geom, end_point.geom,
        points_by_network, networks,
        search_radius_network, increase_radius_network, search_radius_other_networks
    )
    path_wkt = create_multilinestring_from_points(path)
    insert_or_update_result(engine, start_point.geom.wkt, end_point.geom.wkt, path_wkt, total_distance)

    # Print the results
    print("Visited Points:")
    for i, point in enumerate(path):
        print(f"{i + 1}: {point}")

    print(f"Total Path Distance: {total_distance:.2f} meters")
