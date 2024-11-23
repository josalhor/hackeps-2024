import geopandas as gpd
import pandas as pd
import random
from shapely.geometry import Point
from sqlalchemy import create_engine
from geopy.distance import geodesic

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

# def group_points_by_line(lines, points):
#     """Group points by the line they belong to."""
#     points_by_line = {}
#     for line in lines.itertuples():
#         line_points = points[points.within(line.geom)]
#         points_by_line[line.Index] = line_points
#     return points_by_line

# for network_name, network in networks.items():
#     random_line = network.sample(1, random_state=43).iloc[0]
#     print(f"Random line ID: {random_line.id}")
#     lines = pd.DataFrame([random_line], columns=["geom"])
#     print("Lines:", lines)
#     points_by_line = group_points_by_line(lines, points_by_network[network_name])
#     print("Points in line:", points_by_line)
#     # print(f"Points in line {random_line.id}: {len(points_by_line[random_line.id])}")
#     exit()

# Select random start and end points
start_point = inicio.sample(1, random_state=42).iloc[0]
end_point = final.sample(1, random_state=42).iloc[0]

print(f"Start point ID: {start_point.id}, End point ID: {end_point.id}")

# Parameters
search_radius = 1000  # in meters
radius_first_search_increment = 1000  # in meters

# This works for points in format 4326 not 25830
# def haversine_distance(point1, point2):
#     """Calculate the haversine distance (meters) between two points."""
#     return geodesic((point1.y, point1.x), (point2.y, point2.x)).meters

# This works for points in format 25830
def haversine_distance(point1, point2):
    """Calculate the haversine distance (meters) between two points."""
    return point1.distance(point2)


def get_adjacent_points_and_lines(point, points_by_network, red_lines, current_red, radius):
    """
    Find all candidates within the search radius, including:
    - Points from the same or other networks within the radius.
    - Adjacent points in the same network.
    """
    print(f"Finding adjacent points for {point} within {radius} meters...")
    # Find points in other networks within the radius
    # nearby_points = red_points[red_points.distance(point) <= radius]
    nearby_points = pd.DataFrame()
    for network_name, network in points_by_network.items():
        if network_name != current_red:
            new_candidates = network[network.distance(point) <= radius].copy()
            new_candidates["network"] = network_name
            nearby_points = pd.concat([nearby_points, new_candidates])
    
    # Find adjacent points along the same red
    adjacent_points = []
    for line in red_lines[current_red].itertuples():
        if line.geom.distance(point) <= radius:
            # Extract points (start and end) of the line segment
            coords = list(line.geom.coords)
            for i, coord in enumerate(coords):
                candidate = Point(coord)
                if point.distance(candidate) <= radius:
                    adjacent_points.append(candidate)
                # Add next or previous points in the sequence
                if i > 0:
                    adjacent_points.append(Point(coords[i - 1]))
                if i < len(coords) - 1:
                    adjacent_points.append(Point(coords[i + 1]))
    
    # Combine and return unique points
    return gpd.GeoDataFrame(
        geometry=list(set(nearby_points["geom"].tolist() + adjacent_points))
    )

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

def best_first_search(start, end, points_by_network, networks, radius, radius_first_search_increment):
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
    visited.append(real_start)
    current_point = real_start
    current_red = start_network
    
    print('Starting search on network:', current_red)
    while haversine_distance(current_point, real_end) > radius:
        visited.append(current_point)
        
        # Get all candidates (nearby points and adjacent points in the same network)
        candidates = get_adjacent_points_and_lines(
            current_point,
            points_by_network,
            networks,
            current_red,
            radius
        )
        
        if candidates.empty:
            print("No path found!")
            return visited, total_distance
        
        # Calculate heuristic for each candidate
        candidates["heuristic"] = candidates["geometry"].apply(
            lambda pt: haversine_distance(pt, end)
        )
        
        # Choose the best candidate based on heuristic
        best_candidate = candidates.sort_values("heuristic").iloc[0].geometry
        
        # Update total distance
        total_distance += haversine_distance(current_point, best_candidate)
        current_point = best_candidate
    
    # Add the end point to the path
    visited.append(real_end)
    visited.append(end)
    total_distance += haversine_distance(current_point, end)
    
    print("Path found!")
    return visited, total_distance

if __name__ == "__main__":
    print("Start point:", start_point)
    print("End point:", end_point)
    # Run Best-First Search
    path, total_distance = best_first_search(
        start_point.geom, end_point.geom, points_by_network, networks, search_radius, radius_first_search_increment
    )

    # Print the results
    print("Visited Points:")
    for i, point in enumerate(path):
        print(f"{i + 1}: {point}")

    print(f"Total Path Distance: {total_distance:.2f} meters")
