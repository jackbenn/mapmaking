import argparse

import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from geopy.distance import geodesic
from geopy.point import Point
import numpy as np
import logging
from numpy import sin, cos, arccos, arctan2, radians, degrees

logging.basicConfig(level=logging.INFO)

def calculate_normal_vector(point1, point2):
    """Calculate normalized normal vector to great circle containing two points."""
    v1 = np.array(
        [
            cos(radians(point1.latitude)) * cos(radians(point1.longitude)),
            cos(radians(point1.latitude)) * sin(radians(point1.longitude)),
            sin(radians(point1.latitude)),
        ]
    )

    v2 = np.array(
        [
            cos(radians(point2.latitude)) * cos(radians(point2.longitude)),
            cos(radians(point2.latitude)) * sin(radians(point2.longitude)),
            sin(radians(point2.latitude)),
        ]
    )

    normal = np.cross(v2, v1)
    normal /= np.linalg.norm(normal)
    return normal


def calculate_point_with_highest_latitude_point(point1, point2):
    """Calculate the point with the highest latitude along the great circle
    defined by two point1 and point2."""

    normal = calculate_normal_vector(point1, point2)

    lat_highest = degrees(arccos(normal[2]))

    projected_vector = np.array([normal[0], normal[1], 0])
    projected_vector /= np.linalg.norm(projected_vector)

    lon_highest = degrees(arctan2(-projected_vector[1], -projected_vector[0])) % 360

    return Point(lat_highest, lon_highest)


def calculate_initial_bearing(point1, point2):
    """Calculate the initial bearing from point1 to point2."""
    lat1, lon1 = radians(point1.latitude), radians(point1.longitude)
    lat2, lon2 = radians(point2.latitude), radians(point2.longitude)

    dLon = lon2 - lon1
    x = sin(dLon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - (sin(lat1) * cos(lat2) * cos(dLon))
    initial_bearing = arctan2(x, y)

    return (degrees(initial_bearing) + 360) % 360


def calculate_position_on_great_circle(point1, point2, position=0.5):
    """Calculate a point along of a great circle path
    between point1 and point2.
    position=0 gives point1, position=1 gives point2, and 0.5 gives the midpoint."""
    d = geodesic()
    bearing = calculate_initial_bearing(point1, point2)
    dist = d.measure(point1, point2) * position
    return d.destination(point1, bearing=bearing, distance=dist)


def calculate_position_normal_to_great_circle(point1, point2, position=0.5):
    """
    Calculate a position normal (perpendicular) to the great circle path.

    :param point1: Starting location
    :param point2: Ending location
    :param position: Position along the great circle (0=start, 1=end, 0.5=midpoint)
    """
    gc_bearing = (calculate_initial_bearing(point1, point2) + 90) % 360

    d = geodesic()
    dist = d.measure(point1, point2) * position
    normal_point = d.destination(point1, bearing=gc_bearing, distance=dist)

    return normal_point


def calculate_azimuth(point1, point2):
    """Calculate the azimuth from point1 to point2."""
    lat1, lon1 = radians(point1.latitude), radians(point1.longitude)
    lat2, lon2 = radians(point2.latitude), radians(point2.longitude)

    dLon = lon2 - lon1
    x = sin(dLon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - (sin(lat1) * cos(lat2) * cos(dLon))
    initial_bearing = arctan2(x, y)

    return degrees(initial_bearing)  # + 360) % 360


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="map.py",
        description="Generate a map from a long strip between two points.",
    )

    # shrinking this (and zooming in) too much will cause
    # the ocean not to display and the code to slow down
    parser.add_argument(
        "-b",
        "--border",
        default=0.20,
        type=float,
        help="border around great circle, as a fraction of the distance between points.",
    )
    args = parser.parse_args()

    logging.info(f"{args.border=}")
    seattle = Point(47.6062, -122.3321)
    tokyo = Point(35.6895, 139.6917)

    highpoint = calculate_point_with_highest_latitude_point(seattle, tokyo)
    logging.info(f"{highpoint=}")

    azimuth = calculate_azimuth(highpoint, seattle)
    logging.info(f"{azimuth=}")

    projection = ccrs.ObliqueMercator(
        central_longitude=highpoint.longitude,
        central_latitude=highpoint.latitude,
        azimuth=azimuth,
        scale_factor=1,
    )

    fig = plt.figure(figsize=(20, 5))  # Adjust the figure size as needed
    ax = fig.add_subplot(1, 1, 1, projection=projection)

    #ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")


    ax.plot(
        [seattle.longitude, tokyo.longitude],
        [seattle.latitude, tokyo.latitude],
        color="red",
        linewidth=2,
        marker="o",
        transform=ccrs.Geodetic(),
    )

    ax.scatter(
        [highpoint.longitude],
        [highpoint.latitude],
        color="green",
        marker="o",
        s=100,
        transform=ccrs.Geodetic(),
    )

    ax.set_global()


    left = calculate_position_on_great_circle(seattle, tokyo, -args.border)
    right = calculate_position_on_great_circle(seattle, tokyo, 1 + args.border)
    top = calculate_position_normal_to_great_circle(seattle, tokyo, args.border)
    bottom = calculate_position_normal_to_great_circle(seattle, tokyo, -args.border)

    transform = ccrs.Geodetic()
    left_transformed = ax.projection.transform_point(
        left.longitude, left.latitude, transform
    )
    right_transformed = ax.projection.transform_point(
        right.longitude, right.latitude, transform
    )
    top_transformed = ax.projection.transform_point(
        top.longitude, top.latitude, transform
    )
    bottom_transformed = ax.projection.transform_point(
        bottom.longitude, bottom.latitude, transform
    )

    ax.set_extent(
        [
            left_transformed[0],
            right_transformed[0],
            bottom_transformed[1],
            top_transformed[1],
        ],
        ax.projection,
    )

    plt.show()
