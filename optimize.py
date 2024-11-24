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

# Parameters
SEARCH_RADIUS = 1000  # in meters
INCREASE_RADIUS = 1000  # in meters
SEARCH_OTHER_NETWORKS = 500  # in meters
MAX_SEARCH_RADIUS = 7000  # in meters
FINAL_RADIUS = 2000  # in meters

COSTS = {
    "red1": 3,
    "red2": 5,
    "red3": 10
}

# This works for points in format 25830
def haversine_distance(point1, point2):
    """Calculate the haversine distance (meters) between two points."""
    return point1.distance(point2)


def get_adjacent_points_and_lines(point, endpoint, points_by_network, red_lines, current_red, search_radius_network, increase_radius_network, search_radius_other_networks):
    points_current_network = pd.DataFrame()
    heuristic_current_point = haversine_distance(point, endpoint)
    while points_current_network.empty and search_radius_network <= MAX_SEARCH_RADIUS:
        points_radius = points_by_network[current_red][points_by_network[current_red].distance(point) <= search_radius_network].copy()
        search_radius_network += increase_radius_network
        points_current_network = points_radius[points_radius['geom'].apply(lambda pt: haversine_distance(pt, endpoint) < heuristic_current_point)]
    points_current_network = points_radius
    points_current_network['network'] = current_red

    # Filter those with a heuristic less than the current point

    all_points = points_current_network

    # Find points in other networks within the radius
    radius = search_radius_other_networks
    while all_points.empty and radius <= MAX_SEARCH_RADIUS:
        for network_name, network in points_by_network.items():
            if network_name != current_red:
                nearby_points = network[network.distance(point) <= radius].copy()
                nearby_points['network'] = network_name
                all_points = pd.concat([all_points, nearby_points])
        radius += increase_radius_network
    
    return all_points

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


def a_star_search(start, end, points_by_network, networks, search_radius_network, increase_radius_network, search_radius_other_networks):
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
    g_cost = {real_start: 0}  # Cost from start to the current node
    heapq.heappush(priority_queue, (haversine_distance(real_start, real_end), real_start, start_network))  # (priority, point, network)
    came_from = {}  # Track the path
    
    print('Starting search on network:', start_network)
    while priority_queue:
        # Get the point with the smallest f_cost
        current_heuristic, current_point, current_network = heapq.heappop(priority_queue)
        print(f"Current Heuristic: {current_heuristic}")
        current_point = current_point  # Ensure it's hashable
        
        # Check if we reached the goal
        if haversine_distance(current_point, real_end) <= FINAL_RADIUS:
            print("Goal reached!")
            path = reconstruct_path(came_from, current_point)
            path = [start] + path + [real_end, end]
            total_distance = g_cost[real_end] + haversine_distance(real_end, end) + haversine_distance(start, real_start)
            return path, total_distance
        
        # Get candidates
        candidates = get_adjacent_points_and_lines(
            current_point,
            real_end,
            points_by_network,
            networks,
            current_network,
            search_radius_network, increase_radius_network, search_radius_other_networks
        )
        
        if candidates.empty:
            # print("No path found!")
            continue
            return None, total_distance
        
        # Process each candidate
        for _, row in candidates.iterrows():
            neighbor = row.geom
            network = row.network
            cost_multiplier = COSTS[network]
            tentative_g_cost = g_cost[current_point] + haversine_distance(current_point, neighbor) * cost_multiplier
            
            # If the neighbor is already evaluated with a lower cost, skip it
            if neighbor in g_cost and tentative_g_cost >= g_cost[neighbor]:
                continue
            
            # Update path and costs
            came_from[neighbor] = current_point
            g_cost[neighbor] = tentative_g_cost
            
            # Push to priority queue
            min_multiplier = min(COSTS.values())
            heapq.heappush(priority_queue, (tentative_g_cost + haversine_distance(neighbor, real_end) * min_multiplier, neighbor, network))
    
    print("No path found!")
    return None, total_distance

def best_first_search(start, end, points_by_network, networks, search_radius_network, increase_radius_network, search_radius_other_networks):
    """Perform Best-First Search."""
    print('Start Best-First Search algorithm')

    # Initialize priority queue
    priority_queue = []
    visited = set()
    heapq.heapify(priority_queue)
    
    # Find the closest points in the network for start and end
    print('Finding closest points in the network...')
    real_start, start_network = find_closest_to_network(start, points_by_network)
    real_end, end_network = find_closest_to_network(end, points_by_network)
    print('Closest start points found:', start, real_start, 'Distance:', haversine_distance(start, real_start))
    print('Closest end points found:', end, real_end, 'Distance:', haversine_distance(end, real_end))
    
    # Initialize the search
    heapq.heappush(priority_queue, (haversine_distance(real_start, real_end), real_start, start_network, 0))  # (heuristic, point, network, cost)
    came_from = {}  # Track the path
    point_to_network = {}
    min_multiplier = min(COSTS.values())
    
    print('Starting search on network:', start_network)
    while priority_queue:
        # Get the point with the smallest heuristic
        current_heuristic, current_point, current_network, current_cost = heapq.heappop(priority_queue)
        if current_point in visited:
            continue
        visited.add(current_point)
        print(f"Current Heuristic: {current_heuristic}")
        cost_multiplier = COSTS[current_network]
        
        # Check if we reached the goal
        if haversine_distance(current_point, real_end) <= FINAL_RADIUS:
            print("Goal reached!")
            path = reconstruct_path(came_from, current_point)
            path = [start] + path + [real_end, end]
            network_path = [point_to_network.get(point, None) for point in path]
            total_distance = current_cost + haversine_distance(start, real_start) + haversine_distance(real_end, end) + haversine_distance(current_point, real_end) * cost_multiplier
            return path, total_distance, network_path
        
        # Get candidates
        candidates = get_adjacent_points_and_lines(
            current_point,
            real_end,
            points_by_network,
            networks,
            current_network,
            search_radius_network, increase_radius_network, search_radius_other_networks
        )
        
        if candidates.empty:
            continue
        
        # Process each candidate
        for _, row in candidates.iterrows():
            neighbor = row.geom
            network = row.network
            
            # If the neighbor is already visited, skip it
            if neighbor in came_from or neighbor == current_point or neighbor in visited:
                continue
            new_distance = current_cost + haversine_distance(current_point, neighbor) * cost_multiplier
            
            # Update path and push to priority queue
            came_from[neighbor] = current_point
            point_to_network[neighbor] = network
            heapq.heappush(priority_queue, (haversine_distance(neighbor, real_end) * min_multiplier, neighbor, network, new_distance))
    
    print("No path found!")
    return None, None

def greedy_search(start, end, points_by_network, networks, search_radius_network, increase_radius_network, search_radius_other_networks):
    """Perform Greedy Search."""
    print('Start Greedy Search algorithm')

    # Find the closest points in the network for start and end
    print('Finding closest points in the network...')
    real_start, start_network = find_closest_to_network(start, points_by_network)
    real_end, end_network = find_closest_to_network(end, points_by_network)
    print('Closest start points found:', start, real_start, 'Distance:', haversine_distance(start, real_start))
    print('Closest end points found:', end, real_end, 'Distance:', haversine_distance(end, real_end))
    
    current_point = real_start
    current_network = start_network
    came_from = {}  # Track the path
    path = [start]  # Initialize path

    print('Starting greedy search...')
    while True:
        # Check if we reached the goal
        if haversine_distance(current_point, real_end) <= FINAL_RADIUS:
            print("Goal reached!")
            path += reconstruct_path(came_from, current_point) + [real_end, end]
            total_distance = sum(
                haversine_distance(path[i], path[i+1]) for i in range(len(path) - 1)
            )
            return path, total_distance
        
        # Get candidates
        candidates = get_adjacent_points_and_lines(
            current_point,
            real_end,
            points_by_network,
            networks,
            current_network,
            search_radius_network, increase_radius_network, search_radius_other_networks
        )
        
        if candidates.empty:
            print("No path found!")
            return None, None

        # Select the best candidate based on heuristic (closest to the end)
        best_candidate = None
        best_heuristic = float('inf')
        
        for _, row in candidates.iterrows():
            neighbor = row.geom
            network = row.network
            heuristic = haversine_distance(neighbor, real_end)
            if heuristic < best_heuristic:
                best_candidate = (neighbor, network)
                best_heuristic = heuristic

        print(f"Best Heuristic: {best_heuristic}")
        # Update path
        if best_candidate is None:
            print("No valid moves from the current point!")
            return None, None
        
        next_point, next_network = best_candidate
        came_from[next_point] = current_point
        path.append(next_point)
        current_point = next_point
        current_network = next_network
    
    print("No path found!")
    return None, None

def hybrid_search(start, end, points_by_network, networks, search_radius_network, increase_radius_network, search_radius_other_networks):
    """Perform Hybrid Search (Greedy + Best-First Search)."""
    print('Start Hybrid Search algorithm')

    # Find the closest points in the network for start and end
    print('Finding closest points in the network...')
    real_start, start_network = find_closest_to_network(start, points_by_network)
    real_end, end_network = find_closest_to_network(end, points_by_network)
    print('Closest start points found:', start, real_start, 'Distance:', haversine_distance(start, real_start))
    print('Closest end points found:', end, real_end, 'Distance:', haversine_distance(end, real_end))
    
    current_point = real_start
    current_network = start_network
    path = [start]  # Initialize the path
    came_from = {}  # Track the path
    priority_queue = []  # Used for Best-First Search fallback

    last_heuristic = haversine_distance(current_point, real_end)
    greedy_mode = True  # Start in greedy mode

    print('Starting hybrid search...')
    while True:
        # Check if we reached the goal
        if haversine_distance(current_point, real_end) <= FINAL_RADIUS:
            print("Goal reached!")
            path += reconstruct_path(came_from, current_point) + [real_end, end]
            total_distance = sum(
                haversine_distance(path[i], path[i + 1]) for i in range(len(path) - 1)
            )
            return path, total_distance

        # Get candidates
        candidates = get_adjacent_points_and_lines(
            current_point,
            real_end,
            points_by_network,
            networks,
            current_network,
            search_radius_network, increase_radius_network, search_radius_other_networks
        )
        
        if candidates.empty:
            print("No valid moves from the current point!")
            if not priority_queue:
                print("No path found!")
                return None, None
            else:
                # Switch fully to Best-First Search
                greedy_mode = False

        if greedy_mode:
            # Greedy Mode: Pick the best candidate that decreases the heuristic
            best_candidate = None
            best_heuristic = float('inf')
            
            for _, row in candidates.iterrows():
                neighbor = row.geom
                network = row.network
                heuristic = haversine_distance(neighbor, real_end)
                if heuristic < best_heuristic:
                    best_candidate = (neighbor, network)
                    best_heuristic = heuristic
            
            # If no better candidate is found, switch to Best-First Search
            if best_candidate is None or best_heuristic >= last_heuristic:
                print("Greedy mode stuck. Switching to Best-First Search...")
                greedy_mode = False
                continue
            
            # Update path and move to the best candidate
            next_point, next_network = best_candidate
            came_from[next_point] = current_point
            path.append(next_point)
            current_point = next_point
            current_network = next_network
            last_heuristic = best_heuristic

        else:
            # Best-First Search Mode: Expand the search from the priority queue
            if not priority_queue:
                # Add initial point to priority queue if it's empty
                heapq.heappush(priority_queue, (haversine_distance(current_point, real_end), current_point, current_network))
            
            # Process the queue
            current_heuristic, current_point, current_network = heapq.heappop(priority_queue)

            # Get candidates
            candidates = get_adjacent_points_and_lines(
                current_point,
                real_end,
                points_by_network,
                networks,
                current_network,
                search_radius_network, increase_radius_network, search_radius_other_networks
            )
            
            for _, row in candidates.iterrows():
                neighbor = row.geom
                network = row.network
                heuristic = haversine_distance(neighbor, real_end)
                
                # Avoid revisiting
                if neighbor in came_from:
                    continue
                
                # Update path and add to queue
                came_from[neighbor] = current_point
                heapq.heappush(priority_queue, (heuristic, neighbor, network))
            
            # Switch back to greedy mode if there's a promising candidate
            if candidates.empty or all(haversine_distance(row.geom, real_end) >= last_heuristic for _, row in candidates.iterrows()):
                continue
            else:
                print("Switching back to Greedy mode...")
                greedy_mode = True
    
    print("No path found!")
    return None, None


def reconstruct_path(came_from, current_point):
    """Reconstruct the path from came_from mapping."""
    path = [current_point]
    visited = set()
    while current_point in came_from:
        current_point = came_from[current_point]
        if current_point in visited:
            print("Cycle detected!")
            index = path.index(current_point)
            print("Index repeated:", index, 'Current length:', len(path))
            exit()
        visited.add(current_point)
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

def get_coordinates_multiline(multiline):
    """Get the coordinates of a multilinestring."""
    return [
        Point(coord)
        for coords in multiline.geoms
        for coord in get_coordinates(coords)
    ]


def fit_path_to_network(path, networks, network_path):
    lines = []
    complete_gds = pd.concat(networks.values())
    complete_gds.reset_index(drop=True, inplace=True)
    for point in path:
        # Find closest line in the network
        min_line_index = complete_gds.distance(point).idxmin()
        # closest_line = complete_gds.geometry.loc[min_line_index]
        lines.append(min_line_index)
    skipable_rounds = 3
    for _ in range(skipable_rounds):
        skipable = []
        for i in range(len(lines)):
            if i == 0 or i == len(lines) - 1:
                skipable.append(False)
                continue
            is_skipable = lines[i - 1] == lines[i + 1] and lines[i] != lines[i - 1]
            skipable.append(is_skipable)
        new_path = []
        new_lines = []
        new_network_path = []
        for i, point in enumerate(path):
            if skipable[i]:
                continue
            new_path.append(point)
            new_lines.append(lines[i])
            new_network_path.append(network_path[i])
        path = new_path
        lines = new_lines
        network_path = new_network_path
    same_line = []
    for i in range(len(lines) - 1):
        is_same = lines[i] == lines[i + 1]
        same_line.append(is_same)
    same_line.append(False)
    new_path = []
    # Now, we will create a new path by refining it
    # We will start by fitting each point to the closest line
    for i, point in enumerate(path):
        line = complete_gds.geometry.loc[lines[i]]
        points = get_coordinates_multiline(line)
        closest_point = min(points, key=lambda p: p.distance(point))
        new_path.append(closest_point)
    path = new_path
    # Nowe we will add intermediate points to the path
    # When the path stays on the same line
    interpolation_passes = 3
    for _ in range(interpolation_passes):
        new_path = []
        new_same_line = []
        new_lines = []
        new_network_path = []
        for i, point in enumerate(path):
            line = complete_gds.geometry.loc[lines[i]]
            # assert point in line
            assert point.distance(line) < 1e-6, f"Point {point} is not on line {line} distance: {point.distance(line)}"
            new_path.append(point)
            new_same_line.append(same_line[i])
            new_lines.append(lines[i])
            new_network_path.append(network_path[i])
            if same_line[i]:
                points_geometry = get_coordinates_multiline(line)
                # index_current = points_geometry.index(point)
                next_point = path[i + 1]
                interpolated = Point((point.x + next_point.x) / 2, (point.y + next_point.y) / 2)
                # Now we fit the interpolated point to the line
                closest_point = min(points_geometry, key=lambda p: p.distance(interpolated))
                if closest_point != point and closest_point != next_point:
                    new_path.append(closest_point)
                    new_same_line.append(True)
                    new_lines.append(lines[i])
                    new_network_path.append(network_path[i])
        path = new_path
        same_line = new_same_line
        lines = new_lines
        network_path = new_network_path
    total_distance = 0
    for i in range(len(path) - 1):
        distance_next_point = path[i].distance(path[i + 1])
        cost_multiplier = COSTS.get(network_path[i], 1)
        total_distance += distance_next_point * cost_multiplier
    return new_path, total_distance
if __name__ == "__main__":
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

    # print_reslts_table(engine)
    print("Start point:", start_point)
    print("End point:", end_point)
    # ensure_results_table_exists(engine)
    # Run Best-First Search
    path, total_distance, network_path = best_first_search(
        start_point.geom, end_point.geom,
        points_by_network, networks,
        SEARCH_RADIUS, INCREASE_RADIUS, SEARCH_OTHER_NETWORKS
    )
    path, total_distance = fit_path_to_network(path, networks, network_path)
    path_wkt = create_multilinestring_from_points(path)
    ############################
    # SAVE THE RESULTS
    ############################
    # Option 1- Save the results to the database, make sure to run:
    # CREATE TABLE eps.results (
#                 id SERIAL PRIMARY KEY,
#                 start_point GEOMETRY(Point, 25830) NOT NULL,
#                 end_point GEOMETRY(Point, 25830) NOT NULL,
#                 path GEOMETRY(MultiLineString, 25830) NOT NULL,
#                 distance FLOAT NOT NULL,
#                 UNIQUE (start_point, end_point)
    # before running this line
    insert_or_update_result(engine, start_point.geom.wkt, end_point.geom.wkt, path_wkt, total_distance)
    
    # Option 2- Save the results to output.txt
    # Save the results to output.txt
    # with open("output.txt", "w") as f:
    #     f.write("Visited Points:\n")
    #     for i, point in enumerate(path):
    #         f.write(f"{i + 1}: {point}\n")
    #     f.write(f"Total Cost Distance: {total_distance:.2f} meters\n")
    # Create a GeoDataFrame from the path and save it to a shapefile

    # Option 3- Save the results to a shapefile
    # path_gdf = gpd.GeoDataFrame(geometry=path)
    # path_gdf.crs = "EPSG:25830"
    # path_gdf.to_file("path.shp")

    # Print the results
    print("Visited Points:")
    for i, point in enumerate(path):
        print(f"{i + 1}: {point}")

    print(f"Total Cost Distance: {total_distance:.2f} meters")
