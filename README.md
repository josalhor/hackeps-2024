🚗 Andalucía Road Trip Routing Project 🌟
🧭 What’s This About?

This is your very own road trip wizard for Andalucía! It calculates "good enough" routes between two random points in the region, navigating various road types with constraints like costs based on road weight.

You give it some tables, it gives you a route. Who said databases aren’t fun?
🛠️ How It Works

    The Data:
        Start (inicio) and End (final) points in EPSG:4326.
        Roads (red1, red2, red3) with geometries in EPSG:25830.
        Road points (red1_puntos, red2_puntos, red3_puntos) with points of the roads.

    The Output:
        Finds a "good enough" route.
        Saves the route at the database as a MultiLineString.

🧪 Dependencies

This project has more dependencies than a tourist with Google Maps! Make sure you’ve got:

    Data Manipulation:
        geopandas 🗺️ (Spatial operations)
        pandas 🐼 (Table wrangling)

    Geometry Magic:
        shapely ✨ (Geometries and operations)

    Database Wizards:
        sqlalchemy 🏛️ (Database connections)
        geoalchemy2 🌍 (PostGIS extensions)

    Spatial Stuff:
        geopy 🌐 (Distances)

    Others:
        heapq 📚 (Priority queues)
        random 🎲 (To make things spicy)

🎯 How to Use

    Set up a PostgreSQL database with your Andalucía road data.
    Run the script and cross your fingers for a good route.
    Open the resulting path.geojson in your favorite GIS tool to marvel at your newly planned route.

😂 Pro Tips:

    Don’t use this for real-life navigation unless you trust Python more than Google Maps.
    It’s Andalucía—we added a randomizer for the starting point because life’s about the journey, not the destination.

Enjoy your algorithmic viaje! 🌍✨