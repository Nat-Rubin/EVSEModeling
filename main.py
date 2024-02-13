import csv
import sys
from math import acos, sin, cos

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import image as mpimg, transforms
from matplotlib.patches import Circle
import pandas as pd
import re


def main():
    print("Hello, World!")
    coords_m = []
    coords_c = []
    coords_confirmed_chargers = []
    coords_ls = []
    neighborhoods_names = []

    df_pl = pd.read_csv("parking_lots.csv")
    df_m = pd.read_csv("Median Coordinates.csv")
    df_n = pd.read_csv("neighborhoods.csv")
    df_c = pd.read_csv("current_chargers.csv")
    df_confirmed_chargers = pd.read_csv("confirmed_chargers.csv")
    df_ls = pd.read_csv("light_stations.csv")

    wkt_n_dict = {}
    for index, row in df_n.iterrows():
        name = row["name"]
        neighborhoods_names.append(name)
        wkt_n_dict[name]: list[tuple] = []
        wkt: list[str] = row["WKT"][10:][:-2]
        wkt = re.sub(r', ', ',', wkt)
        wkt_as_floats = []

        for coords in wkt.split(","):
            for coord in coords.split(" "):
                wkt_as_floats.append(float(coord))

            wkt_n_dict[name].append(tuple(wkt_as_floats))
            wkt_as_floats = []

    for index, row in df_m.iterrows():
        x_coord = row["X / Weights"]
        y_coord = row["Y / Weights"]
        coords_m.append((x_coord, y_coord))

    for index, row in df_c.iterrows():
        x_coord = row["X Coordinate"]
        y_coord = row["Y Coordinate"]
        coords_c.append((x_coord, y_coord))

    for index, row in df_ls.iterrows():
        x_coord = row["X Coordinate"]
        y_coord = row["Y Coordinate"]
        coords_ls.append((x_coord, y_coord))

    distances = distance_to_light_stations(df_confirmed_chargers, df_ls)
    print(distances)
    distances_to_csv(distances)
    graph_plot(df_pl, wkt_n_dict, neighborhoods_names, coords_m, coords_c, coords_ls, df_confirmed_chargers, distances)


def distances_to_csv(distances) -> None:
    headers = ["Charger Name", "Charger Coordinates",
               "Light Station Name", "Light Station Coordinates", "Distance (km)"]
    try:
        with open("distances.csv", "w", newline='') as file:
            print("HI")
            writer = csv.writer(file)
            writer.writerow(headers)
            for key, value in distances.items():
                data = [key, f"{value[0][0]}, {value[0][1]}", value[1], f"{value[2][0]}, {value[2][1]}", value[3]]
                writer.writerow(data)
    except Exception as e:
        print("Error: ", e)


def lon_lat_to_km(coord1: tuple, coord2: tuple) -> float:
    # Uses haversine formula
    lat1 = coord1[0]
    lon1 = coord1[1]
    lat2 = coord2[0]
    lon2 = coord2[1]
    earth_radius = 6371  # km

    distance = acos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon2-lon1)) * earth_radius
    return distance

def distance_to_light_stations(df_confirmed_chargers, df_ls) -> dict:
    distances: dict[str: [tuple, str, tuple, float]] = {}

    name_index = 1
    for index, row in df_confirmed_chargers.iterrows():
        name = row["name"]
        if isinstance(name, float) or isinstance(name, float):
            continue

        x_coord = row["X Coordinates"]
        y_coord = row["Y Coordinates"]
        if not isinstance(x_coord, float) and not isinstance(y_coord, float):
            try:
                x_coord = float(x_coord)
                y_coord = float(y_coord)
            except:
                continue
        else:
            continue
        if name in distances:
            name = f'{name}{name_index}'
            name_index += 1
        distances[name] = [(x_coord, y_coord)]

    ls_info: list[list[str, float, float]] = []
    # TODO: GIVE POINTS UNIQUE NAMES AND ADD FIXES TO GRAPH SECTION
    for index, row in df_ls.iterrows():
        ls_name = row["name"]
        x_coord = float(row["X Coordinate"])
        y_coord = float(row["Y Coordinate"])

        ls_info.append([ls_name, x_coord, y_coord])

    for charger_name in list(distances.keys()):
        smallest_distance = sys.maxsize
        smallest_distance_info = []
        x_coord_charger = distances[charger_name][0][0]
        y_coord_charger = distances[charger_name][0][1]
        for info in ls_info:
            ls_name = info[0]
            x_coord_ls = info[1]
            y_coord_ls = info[2]
            # distance = np.sqrt((x_coord_ls - x_coord_charger) ** 2 + (y_coord_ls - y_coord_charger) ** 2)
            distance = lon_lat_to_km((x_coord_ls, y_coord_ls), (x_coord_charger, y_coord_charger))
            if distance < smallest_distance:
                smallest_distance = distance
                smallest_distance_info = info
        distances[charger_name].append(smallest_distance_info[0])
        distances[charger_name].append((smallest_distance_info[1], smallest_distance_info[2]))
        distances[charger_name].append(smallest_distance)

    return distances


def graph_plot(df_pl, wkt_n_dict, neighborhoods_names, coords_m, coords_c, coords_ls, df_confirmed_chargers, distances: dict) -> None:
    # Boundaries for graph. Changing these will change the image size (change at your own risk!!!)
    plot_parking_lots = True
    plot_neighborhoods = True
    plot_markers = False
    plot_medians = True
    plot_current_chargers = True
    plot_light_stations = True
    plot_confirmed_chargers = True
    plot_distances = True

    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax1 = fig.add_subplot(111)

    img = np.asarray(Image.open("Eilat2.png"))
    height, width, _ = img.shape
    x_min = 34.92
    y_min = 29.522
    const = 27000
    x_max = width / const
    y_max = height / const
    plt.gca().set_aspect(height / width)

    plt.imshow(img, extent=(x_min, x_max + x_min + .007, y_min, y_max + y_min))

    # neighborhoods
    marker_coords = [wkt_n_dict[next(iter(wkt_n_dict.keys()))][0], wkt_n_dict[next(iter(wkt_n_dict.keys()))][0],
                     wkt_n_dict[next(iter(wkt_n_dict.keys()))][0], wkt_n_dict[next(iter(wkt_n_dict.keys()))][0]]
    coords_n = []
    for name in neighborhoods_names:
        for coords in wkt_n_dict[name]:
            if coords[0] < marker_coords[0][0]:
                marker_coords[0] = coords
            elif coords[0] > marker_coords[1][0]:
                marker_coords[1] = coords
            elif coords[1] < marker_coords[2][1]:
                marker_coords[2] = coords
            elif coords[1] > marker_coords[3][1]:
                marker_coords[3] = coords
            coords_n.append(coords)
        x_n = [coord[0] + .00 for coord in coords_n]
        y_n = [coord[1] + .00 for coord in coords_n]

        if plot_neighborhoods:
            ax1.plot(x_n, y_n, color='gray')
            ax1.fill(x_n, y_n, color='lightblue', alpha=0.5)
        coords_n = []

    # current chargers
    if plot_current_chargers:
        x_c = [coord[0] for coord in coords_c]
        y_c = [coord[1] for coord in coords_c]
        ax1.scatter(x_c, y_c, color='orange', s=8)

    # parking lots
    if plot_parking_lots:
        for index, row in df_pl.iterrows():
            colors = ["lightblue", "blue", "darkblue"]
            weight = row["Weight"]
            color = colors[weight - 1]
            x_coord = row["X"]
            y_coord = row["Y"]
            ax1.scatter(x_coord, y_coord, s=10 * weight, color=color)

    # confirmed chargers
    if plot_confirmed_chargers:
        for index, row in df_confirmed_chargers.iterrows():
            name = row["name"]
            x_coord = None
            y_coord = None
            try:
                x_coord = float(row["X Coordinates"])
                y_coord = float(row["Y Coordinates"])
            except:
                pass
            ax1.scatter(x_coord, y_coord, color='red', s=10)

    # light stations
    if plot_light_stations:
        x_ls = [coord[0] for coord in coords_ls]
        y_ls = [coord[1] for coord in coords_ls]
        ax1.scatter(x_ls, y_ls, color='yellow', s=10)

    # distances to light stations
    if plot_distances:
        for value in distances.values():
            x_distance = [value[0][0], value[2][0]]
            y_distance = [value[0][1], value[2][1]]
            ax1.plot(x_distance, y_distance, linestyle='-', color='orange')

    # median coords
    if plot_medians:
        x_m = [coord[0] for coord in coords_m]
        y_m = [coord[1] for coord in coords_m]
        ax1.scatter(x_m, y_m, color='red', s=8)

    # boarder coords/markers for helping with image
    if plot_markers:
        x_n = [coord[0] + .00 for coord in marker_coords]
        y_n = [coord[1] + .00 for coord in marker_coords]
        ax1.scatter(x_n, y_n, color='yellow', s=8)

    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.title("Parking Lots and Chargers")

    plt.grid(False)
    plt.show()


if __name__ == '__main__':
    main()
