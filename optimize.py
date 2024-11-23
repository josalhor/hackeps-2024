import geopandas as gpd
import pandas as pd
import random
from shapely.geometry import Point, MultiLineString, Point
from shapely.ops import nearest_points
from sqlalchemy import create_engine
from geopy.distance import geodesic
from sqlalchemy import inspect
from geoalchemy2 import Geometry
from sqlalchemy.sql import text
import heapq

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

def extract_network_points(network):
    """Efficiently extracts points from network geometries and returns a GeoDataFrame with points and geometry_id."""
    geometry_ids = []
    points = []

    # Flatten extraction using direct list appending
    for geometry_id, geom in zip(network.index, network['geom']):
        for linestring in geom.geoms:
            for coord in linestring.coords:
                geometry_ids.append(geometry_id)
                points.append(Point(coord))

    # Create a GeoDataFrame from flat lists
    network_points = gpd.GeoDataFrame({'geometry_id': geometry_ids, 'geometry': points}, crs=network.crs)
    return network_points


# Apply the function to each network
import time
start = time.time()
# network_points_red1 = extract_network_points(red1)
# network_points_red2 = extract_network_points(red2)
# network_points_red3 = extract_network_points(red3)
# Save to file
# network_points_red1.to_file("network_points_red1.geojson", driver="GeoJSON")
# network_points_red2.to_file("network_points_red2.geojson", driver="GeoJSON")
# network_points_red3.to_file("network_points_red3.geojson", driver="GeoJSON")
# Load from file
network_points_red1 = gpd.read_file("network_points_red1.geojson")
network_points_red2 = gpd.read_file("network_points_red2.geojson")
network_points_red3 = gpd.read_file("network_points_red3.geojson")
extracted_points = {
    "red1": network_points_red1,
    "red2": network_points_red2,
    "red3": network_points_red3
}

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
increase_radius_network = 500  # in meters
search_radius_other_networks = 200  # in meters
max_search_radius = 4500  # in meters

# This works for points in format 4326 not 25830
# def haversine_distance(point1, point2):
#     """Calculate the haversine distance (meters) between two points."""
#     return geodesic((point1.y, point1.x), (point2.y, point2.x)).meters

# This works for points in format 25830
def haversine_distance(point1, point2):
    """Calculate the haversine distance (meters) between two points."""
    return point1.distance(point2)


def get_adjacent_points_and_lines(point, endpoint, points_by_network, networks, extracted_points, current_red):
    """
    Find all candidates within the search radius, including:
    - Points from the same or other networks within the radius.
    - Adjacent points in the same network.
    """
    # print(f"Finding adjacent points for {point}")
    # Find points in other networks within the radius
    # nearby_points = red_points[red_points.distance(point) <= radius]
    points_current_network = pd.DataFrame()
    radius = search_radius_network
    while len(points_current_network) < 2 and radius <= max_search_radius:
        points_current_network = extracted_points[current_red][extracted_points[current_red].distance(point) <= radius].copy()
        radius += increase_radius_network
    points_current_network['network'] = current_red

    # Filter those with a heuristic less than the current point

    other_networks = pd.DataFrame()
    # Find points in other networks within the radius
    for network_name, network in points_by_network.items():
        radius = search_radius_other_networks
        if network_name != current_red:
            nearby_points = network[network.distance(point) <= radius].copy()
            nearby_points['network'] = network_name
            other_networks = pd.concat([other_networks, nearby_points])
    
    # We have a problem here, points_current_network has 'geometry' and
    # other_networks has 'geom'
    # To deal with this, we will rename the columns
    points_current_network.rename(columns={'geometry': 'geom'}, inplace=True)

    # Combine and return unique points
    return pd.concat([points_current_network, other_networks], ignore_index=True)

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


def a_star_search(start, end, points_by_network, networks, extracted_points):
    """Perform A* Search."""
    print('Start A* algorithm')

    # Initialize priority queue
    priority_queue = []
    heapq.heapify(priority_queue)
    
    # Find the closest points in the network for start and end
    print('Finding closest points in the network...')
    real_start, start_network = find_closest_to_network(start, points_by_network)
    real_end, end_network = find_closest_to_network(end, points_by_network)
    print('Closest start points found:', start, real_start, 'Distance:', haversine_distance(start, real_start))
    print('Closest end points found:', end, real_end, 'Distance:', haversine_distance(end, real_end))
    
    # Initialize the search
    g_cost = {real_start: haversine_distance(start, real_start)}  # Cost from start to point
    heapq.heappush(priority_queue, (haversine_distance(real_start, real_end), real_start, start_network))  # (priority, point, network)
    came_from = {}  # Track the path
    
    print('Starting search on network:', start_network)
    while priority_queue:
        # Get the point with the smallest f_cost
        current_heuristic, current_point, current_network = heapq.heappop(priority_queue)
        print(f'Current Heuristic: {current_heuristic}')
        assert isinstance(current_point, Point), f"Current point is not a Point: {current_point}"
        
        # Check if we reached the goal
        if haversine_distance(current_point, real_end) <= search_radius_network:
            print("Goal reached!")
            path = reconstruct_path(came_from, current_point)
            path = [start] + path + [end]
            total_distance = g_cost[current_point] + haversine_distance(real_end, end)
            return path, total_distance
        
        # Get candidates
        candidates = get_adjacent_points_and_lines(
            current_point,
            real_end,
            points_by_network,
            networks,
            extracted_points,
            current_network
        )
        
        if candidates.empty:
            # print("No path found!")
            # Backtrack to the closest point in the network
            continue
        
        # Process each candidate
        for _, row in candidates.iterrows():
            neighbor = row.geom
            assert isinstance(neighbor, Point), f"Neighbor is not a Point: {neighbor}"
            network = row.network
            tentative_g_cost = g_cost[current_point] + haversine_distance(current_point, neighbor)
            
            # If the neighbor is already evaluated with a lower cost, skip it
            if neighbor in g_cost and tentative_g_cost >= g_cost[neighbor]:
                continue
            
            # Update path and costs
            came_from[neighbor] = current_point
            assert not isinstance(current_point, MultiLineString), f"Current point is a MultiLineString: {current_point}"
            g_cost[neighbor] = tentative_g_cost
            
            # Push to priority queue
            heapq.heappush(priority_queue, (tentative_g_cost + haversine_distance(neighbor, real_end), neighbor, network))
    
    raise ValueError("No path found!")


def reconstruct_path(came_from, current_point):
    """Reconstruct the path from came_from mapping."""
    path = [current_point]
    assert not isinstance(current_point, MultiLineString), f"Current point is a MultiLineString: {current_point}"
    while current_point in came_from:
        current_point = came_from[current_point]
        assert not isinstance(current_point, MultiLineString), f"Current point is a MultiLineString: {current_point}"
        path.append(current_point)
    path.reverse()
    return path

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
    path, total_distance = a_star_search(
        start_point.geom, end_point.geom,
        points_by_network, networks, extracted_points
    )
    print('Saving results...')
    path_wkt = create_multilinestring_from_points(path)
    insert_or_update_result(engine, start_point.geom.wkt, end_point.geom.wkt, path_wkt, total_distance)

    # Print the results
    print("Visited Points:")
    for i, point in enumerate(path):
        print(f"{i + 1}: {point}")

    print(f"Total Path Distance: {total_distance:.2f} meters")
