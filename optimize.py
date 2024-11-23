import geopandas as gpd
import pandas as pd
import random
from shapely.geometry import Point, MultiLineString, Point
from shapely.ops import nearest_points
from shapely import get_coordinates
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

class Neighbour:
    def __init__(self, point, reach_path, network, line, real_distance, heuristic_distance):
        self.point = point
        self.reach_path = reach_path
        self.network = network
        self.line = line
        assert line.geom_type == "MultiLineString", f"Line is not a MultiLineString: {line}"
        self.real_distance = real_distance
        self.heuristic_distance = heuristic_distance
    def __lt__(self, other):
        if self.heuristic_distance == other.heuristic_distance:
            return self.real_distance < other.real_distance
        return self.heuristic_distance < other.heuristic_distance

class SpatialGraph:
    def __init__(self, networks, points_by_network):
        self.networks = networks
        self.points_by_network = points_by_network

    def find_closest_line(self, point, network):
        """Find the closest multilinestring in the specified network to the given point."""
        multilines = self.networks[network]
        distances = multilines.distance(point)
        closest_index = distances.idxmin()
        return multilines.loc[closest_index].geom, distances[closest_index]

    def find_closest_lines(self, point, network, search_radius):
        """Find all multilinestrings within the search radius of the given point."""
        multilines = self.networks[network]
        distances = multilines.distance(point)
        close_lines = multilines[distances <= search_radius]
        return close_lines

    def get_coordinates(self, multiline):
        """Get the coordinates of a multilinestring."""
        return [
            Point(coord)
            for coords in multiline.geoms
            for coord in get_coordinates(coords)
        ]
    
    def get_start_end_points(self, multiline):
        """Get the start and end points of a multilinestring."""
        coords = self.get_coordinates(multiline)
        return coords[0], coords[-1]

    def find_closest_point_line(self, point, multiline):
        """Find the closest start/end point of the multilinestring to the given point."""
        points = self.get_coordinates(multiline)
        distances = [point.distance(pt) for pt in points]
        closest_index = distances.index(min(distances))
        return points[closest_index]
    
    def find_closest_point_not_in_network(self, point):
        """ Find the closest point in the network to the given point. """
        min_distance = None
        min_point = None
        min_line = None
        for network_name, network in self.points_by_network.items():
            # We will use find_closest_line and find_closest_point_line
            closest_line, _ = self.find_closest_line(point, network_name)
            closest_point = self.find_closest_point_line(point, closest_line)
            distance = closest_point.distance(point)
            if min_distance is None or distance < min_distance:
                min_distance = distance
                min_point = closest_point
                min_line = closest_line

        return min_point, network_name, min_line
    
    def find_closest_point_in_network(self, point, network):
        """Find the closest point in the network to the given point."""
        candidates = self.points_by_network[network]
        closest_point = candidates.loc[
            candidates.distance(point).idxmin()
        ]
        return closest_point.geom

    def find_neighbours(self, neighbour:Neighbour, last_point):
        """Find the neighbours of a point in a multilinestring and across networks."""
        multiline = neighbour.line
        network = neighbour.network
        point = neighbour.point
        # Now we are sure that the point is in the 
        coords = self.get_coordinates(multiline)
        assert point in coords, f"Point {point} not in the line {coords}"
        index_point = coords.index(point)
        
        neighbours = []
        jump_to = set()
        # We will do two loops, one backwards and one forwards
        def evaluate_neighbour(i, reach, acc_distance):
            for network_name in self.networks:
                if network_name == network:
                    continue
                if network_name in jump_to:
                    continue
                # Find the closest line in the other network
                closest_jump = self.find_closest_point_in_network(coords[i], network_name)
                distance_jump = closest_jump.distance(coords[i])
                if distance_jump > 500:
                    continue
                closest_line, distance_line = self.find_closest_line(coords[i], network_name)
                if distance_line > 500:
                    continue
                closest_point = self.find_closest_point_line(coords[i], closest_line)
                distance = closest_point.distance(coords[i])
                if distance <= 500:
                    # We can add this point
                    new_neighbour = Neighbour(closest_point, reach, network_name, closest_line, neighbour.real_distance + acc_distance, haversine_distance(closest_point, last_point))
                    neighbours.append(new_neighbour)
                    jump_to.add(network_name)
                

        reach = []
        acc_distance = 0
        for i in range(index_point - 1, -1, -1):
            reach.append(coords[i])
            acc_distance += coords[i].distance(coords[i + 1]) 
            evaluate_neighbour(i, reach, acc_distance)
        # We also need to evaluate the start point
        if index_point != 0:
            for close_line in self.find_closest_lines(coords[0], network, 500).geom:
                closest_point = self.find_closest_point_line(coords[0], close_line)
                distance = closest_point.distance(coords[0])
                if distance <= 500 and closest_point != coords[0]:
                    new_neighbour = Neighbour(closest_point, reach, network, close_line, neighbour.real_distance + acc_distance, haversine_distance(closest_point, last_point))
                    neighbours.append(new_neighbour)

        reach = []
        acc_distance = 0
        for i in range(index_point + 1, len(coords)):
            reach.append(coords[i])
            acc_distance += coords[i].distance(coords[i - 1])
            evaluate_neighbour(i, reach, acc_distance)
        # We also need to evaluate the end point
        if index_point != len(coords) - 1:
            for close_line in self.find_closest_lines(coords[-1], network, 500).geom:
                closest_point = self.find_closest_point_line(coords[-1], close_line)
                distance = closest_point.distance(coords[-1])
                if distance <= 500 and closest_point != coords[-1]:
                    new_neighbour = Neighbour(closest_point, reach, network, close_line, neighbour.real_distance + acc_distance, haversine_distance(closest_point, last_point))
                    neighbours.append(new_neighbour)

        return neighbours

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


def a_star_search(start, end, spatial_graph:SpatialGraph, search_radius_network, increase_radius_network, search_radius_other_networks):
    """Perform A* Search."""
    total_distance = 0  # Track total distance
    print('Start A* algorithm')

    # Initialize priority queue
    priority_queue = []
    heapq.heapify(priority_queue)
    
    # Find the closest points in the network for start and end
    print('Finding closest points in the network...')
    real_start, start_network, start_line = spatial_graph.find_closest_point_not_in_network(start)
    real_end, _, _ = spatial_graph.find_closest_point_not_in_network(end)
    print('Closest start points found:', start, real_start, 'Distance:', haversine_distance(start, real_start))
    print('Closest end points found:', end, real_end, 'Distance:', haversine_distance(end, real_end))
    
    start_heuristic = haversine_distance(real_start, real_end)
    # Initialize the search
    g_cost = {real_start: 0}  # Cost from start to the current node
    neighbor = Neighbour(real_start, [], start_network, start_line, 0, start_heuristic)
    heapq.heappush(priority_queue, (start_heuristic, neighbor))
    ### DEBUG
    # print(spatial_graph.find_neighbours(neighbor, real_end))
    # exit()
    came_from = {}  # Track the path
    
    print('Starting search on network:', start_network)
    while priority_queue:
        # Get the point with the smallest f_cost
        _, neighbor = heapq.heappop(priority_queue)
        print('Visiting point:', neighbor.point)
        print('Number of points line:', len(spatial_graph.get_coordinates(neighbor.line)))
        print(f'Current heuristic distance: {neighbor.heuristic_distance:.2f} current point: {neighbor.point}')
        neighbor : Neighbour = neighbor # type hinting
        
        # Check if we reached the goal
        if haversine_distance(neighbor.point, real_end) <= search_radius_network:
            print("Goal reached!")
            path = reconstruct_path(came_from, neighbor)
            path.extend([real_end, end])  # Append the final points
            total_distance = haversine_distance(start, real_start) + neighbor.real_distance + haversine_distance(real_end, end)
            return path, total_distance
        
        # Get candidates
        candidates = spatial_graph.find_neighbours(neighbor, real_end)
        
        if not candidates:
            # print("No path found!")
            # Found a dead end, backtrack
            continue
        
        # Process each candidate
        for new_neighbor in candidates:
            new_neighbor : Neighbour = new_neighbor # type hinting
            tentative_g_cost = new_neighbor.real_distance
            
            # If the neighbor is already evaluated with a lower cost, skip it
            if new_neighbor.point in g_cost and tentative_g_cost >= g_cost[new_neighbor.point]:
                continue
            # if new_neighbor.point in g_cost:
            #     print(f"Found a better path to {new_neighbor.point} with cost {tentative_g_cost} instead of {g_cost[new_neighbor.point]}")
            # else:
            #     print(f"Found a new path to {new_neighbor.point} with cost {tentative_g_cost}")
            # Update path and costs
            came_from[new_neighbor.point] = neighbor
            g_cost[new_neighbor.point] = new_neighbor.real_distance
            
            # Push to priority queue
            heapq.heappush(priority_queue, (new_neighbor.heuristic_distance, new_neighbor))
    
    print("No path found!")
    return None, total_distance


def reconstruct_path(came_from, neighbor:Neighbour):
    """Reconstruct the path from came_from mapping."""
    path = [neighbor.point]
    for c in neighbor.reach_path[::-1]:
        path.append(c)
    while neighbor.point in came_from:
        neighbor = came_from[neighbor.point]
        path.append(neighbor.point)
        for c in neighbor.reach_path[::-1]:
            path.append(c)
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
    spatial_graph = SpatialGraph(networks, points_by_network)
    path, total_distance = a_star_search(
        start_point.geom, end_point.geom,
        spatial_graph,
        search_radius_network, increase_radius_network, search_radius_other_networks
    )
    path_wkt = create_multilinestring_from_points(path)
    insert_or_update_result(engine, start_point.geom.wkt, end_point.geom.wkt, path_wkt, total_distance)

    # Print the results
    print("Visited Points:")
    for i, point in enumerate(path):
        print(f"{i + 1}: {point}")

    print(f"Total Path Distance: {total_distance:.2f} meters")
