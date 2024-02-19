import csv
import math
import sys
from math import acos, sin, cos

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import image as mpimg, transforms
from matplotlib.patches import Circle
import pandas as pd
from pandas import DataFrame
import re


def main():
    print("Hello, World!")
    coords_m = []
    coords_c = []
    coords_confirmed_chargers = []
    coords_ls = []
    neighborhoods_names = []

    file_stage_1 = "Stage 1 (Gas Station).csv"
    file_stage_2 = "Stage 2 (Shopping Centers).csv"
    file_stage_3 = "Stage 3 (Municipal).csv"
    file_stage_4 = "Stage 4 (Residential).csv"
    file_all_stages = "All Stages.csv"
    file_all_stages_header = ["WKT", "name", "Neighborhood", "Spots", "Slow", "Medium", "Fast", "Ultra Fast", "kW",
                              "X Coordinate", "Y Coordinate"]
    stages = [file_stage_1, file_stage_2, file_stage_3, file_stage_4]
    rows = []
    for stage in stages:
        with open(stage, newline='') as f:
            reader = csv.reader(f)
            next(reader)
            rows.extend(reader)

    with open(file_all_stages, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(file_all_stages_header)
        writer.writerows(rows)

    df_pl = pd.read_csv("parking_lots.csv")
    df_m = pd.read_csv("Median Coordinates.csv")
    df_n = pd.read_csv("neighborhoods.csv")
    df_c = pd.read_csv("current_chargers.csv")
    df_stage_1 = pd.read_csv(file_stage_1)
    df_stage_2 = pd.read_csv(file_stage_2)
    df_stage_3 = pd.read_csv(file_stage_3)
    df_stage_4 = pd.read_csv(file_stage_4)
    df_confirmed_chargers = pd.read_csv(file_all_stages)
    df_ls = pd.read_csv("light_stations.csv")

    df_stage_list = [df_stage_1, df_stage_2, df_stage_3, df_stage_4, df_confirmed_chargers, df_ls]

    print(type(df_stage_1))

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
    distances_to_csv(distances)
    # num_chargers = count_chargers(df_confirmed_chargers)  # broken
    # print(num_chargers)
    graph_plot(df_pl, wkt_n_dict, neighborhoods_names, coords_m, coords_c, coords_ls, df_stage_list, distances)


# Doesn't work because csv file is so messed up that some points don't have any chargers listed in them
# Also, probably don't need this
def count_chargers(df_confirmed_chargers) -> tuple:
    num_medium = 0
    num_fast = 0
    num_ultra_fast = 0
    for index, row in df_confirmed_chargers.iterrows():
        name = row["name"]
        if isinstance(name, float):
            continue

        if not isinstance(row["Medium"], float):
            num_medium += int(row["Medium"])
        if not isinstance(row["Fast"], float):
            num_fast += int(row["Fast"])
        if not isinstance(row["Ultra Fast"], float):
            num_ultra_fast += int(row["Ultra Fast"])

    return num_medium, num_fast, num_ultra_fast


def distances_to_csv(distances) -> None:
    headers = ["Charger Name", "Charger Coordinates",
               "Light Station Name", "Light Station Coordinates", "Distance (km)"]
    try:
        with open("distances.csv", "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            for key, value in distances.items():
                data = [key, f"{value[0][0]}, {value[0][1]}", value[1], f"{value[2][0]}, {value[2][1]}", value[3]]
                writer.writerow(data)
    except Exception as e:
        print("Error: ", e)


# Haversine Formula
def lon_lat_to_km(coord1: tuple, coord2: tuple) -> float:
    # Uses haversine formula
    lat1 = coord1[0]
    lon1 = coord1[1]
    lat2 = coord2[0]
    lon2 = coord2[1]
    earth_radius = 6371  # km
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    distance = acos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon2 - lon1)) * earth_radius
    # print("distance: ", distance)
    return distance


def distance_to_light_stations(df_confirmed_chargers, df_ls) -> dict:
    distances: dict[str: [tuple, str, tuple, float]] = {}

    name_index = 1
    for index, row in df_confirmed_chargers.iterrows():
        name = row["name"]
        x_coord = float(row["X Coordinate"])
        y_coord = float(row["Y Coordinate"])

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

        ls_info.append([ls_name, x_coord, y_coord])  # idk why it doesn't like this but it works anyway

    for charger_name in list(distances.keys()):
        smallest_distance = sys.maxsize
        smallest_distance_info = []
        x_coord_charger = distances[charger_name][0][0]
        y_coord_charger = distances[charger_name][0][1]
        for info in ls_info:
            ls_name = info[0]
            x_coord_ls = info[1]
            y_coord_ls = info[2]
            distance = lon_lat_to_km((x_coord_ls, y_coord_ls), (x_coord_charger, y_coord_charger))
            if distance < smallest_distance:
                smallest_distance = distance
                smallest_distance_info = info
        if charger_name == "Residential2":
            print(smallest_distance_info)
        distances[charger_name].append(smallest_distance_info[0])
        distances[charger_name].append((smallest_distance_info[1], smallest_distance_info[2]))
        distances[charger_name].append(smallest_distance)

    return distances


# PLOTTING FUNCTIONS


def plot_parking_lots_func(ax, df_parking_lots):
    pass
    colors = ["lightblue", "blue", "darkblue"]
    small_legend: bool = False
    medium_legend: bool = False
    large_legend: bool = False
    for index, row in df_parking_lots.iterrows():
        weight = row["Weight"]
        base_dot_size = 15
        color = colors[weight - 1]
        x_coord = row["X"]
        y_coord = row["Y"]
        if weight == 1 and not small_legend:
            ax.scatter(x_coord, y_coord, s=base_dot_size * weight, color=colors[0], label="small")
            small_legend = True
        elif weight == 2 and not medium_legend and small_legend:
            ax.scatter(x_coord, y_coord, s=base_dot_size * weight, color=colors[1], label="medium")
            medium_legend = True
        elif weight == 3 and not large_legend and small_legend and medium_legend:
            ax.scatter(x_coord, y_coord, s=base_dot_size * weight, color=colors[2], label="large")
            large_legend = True
        else:
            ax.scatter(x_coord, y_coord, s=base_dot_size * weight, color=color)


def plot_current_chargers_func(ax, coords_c, color='magenta', s=20):
    x_c = [coord[0] for coord in coords_c]
    y_c = [coord[1] for coord in coords_c]
    ax.scatter(x_c, y_c, color=color, s=s, label="stage 0 chargers")


def plot_stage_1_func(ax, df_stage_1, color='#00FF00', s=20):
    legend: bool = False
    for index, row in df_stage_1.iterrows():
        name = row["name"]
        x_coord = float(row["X Coordinate"])
        y_coord = float(row["Y Coordinate"])

        if not legend:
            ax.scatter(x_coord, y_coord, c=color, s=s, label="stage 1 chargers")
            legend = True
        else:
            ax.scatter(x_coord, y_coord, c=color, s=s)


def plot_stage_2_func(ax, df_stage_2, color='blue', s=20):
    legend: bool = False
    for index, row in df_stage_2.iterrows():
        name = row["name"]
        x_coord = float(row["X Coordinate"])
        y_coord = float(row["Y Coordinate"])

        if not legend:
            ax.scatter(x_coord, y_coord, color=color, s=s, label="stage 2 chargers")
            legend = True
        else:
            ax.scatter(x_coord, y_coord, color=color, s=s)


def plot_stage_3_func(ax, df_stage_3, color='yellow', s=20):
    legend: bool = False
    for index, row in df_stage_3.iterrows():
        name = row["name"]
        x_coord = float(row["X Coordinate"])
        y_coord = float(row["Y Coordinate"])

        if not legend:
            ax.scatter(x_coord, y_coord, color=color, s=s, label="stage 3 chargers")
            legend = True
        else:
            ax.scatter(x_coord, y_coord, color=color, s=s)


def plot_stage_4_func(ax, df_stage_4, color='orange', s=20):
    legend: bool = False
    for index, row in df_stage_4.iterrows():
        name = row["name"]
        x_coord = float(row["X Coordinate"])
        y_coord = float(row["Y Coordinate"])

        if not legend:
            ax.scatter(x_coord, y_coord, color=color, s=s, label="stage 4 chargers")
            legend = True
        else:
            ax.scatter(x_coord, y_coord, color=color, s=s)


def plot_confirmed_chargers_func(ax, df_confirmed_chargers, annotate_confirmed_chargers_flag):
    legend: bool = False
    for index, row in df_confirmed_chargers.iterrows():
        name = row["name"]
        x_coord = float(row["X Coordinate"])
        y_coord = float(row["Y Coordinate"])

        if not legend:
            ax.scatter(x_coord, y_coord, color='green', s=15, label="suitable lots")
            legend = True
        else:
            ax.scatter(x_coord, y_coord, color='green', s=15)
        if annotate_confirmed_chargers_flag:
            print(name)
            print(x_coord, y_coord)
            ax.annotate(name, (x_coord, y_coord), textcoords="offset points", xytext=(0, 2), ha='center', fontsize=4)


def plot_distances_func(ax, distances):
    distances_legend: bool = False
    for value in distances.values():
        x_distance = [value[0][0], value[2][0]]
        y_distance = [value[0][1], value[2][1]]
        if not distances_legend:
            ax.plot(x_distance, y_distance, linestyle='-', color='orange',
                    label="distance from charger to nearest light station")
            distances_legend = True
        else:
            ax.plot(x_distance, y_distance, linestyle='-', color='orange')


def graph_plot(df_pl, wkt_n_dict, neighborhoods_names, coords_m, coords_c, coords_ls, stages: list[DataFrame],
               distances: dict) -> None:
    plot_title = "Stage Four"

    # DATAFRAMES FROM stages PARAM
    df_confirmed_chargers = stages[4]
    df_stage_1 = stages[0]
    df_stage_2 = stages[1]
    df_stage_3 = stages[2]
    df_stage_4 = stages[3]

    # BOOLEANS
    plot_neighborhoods_flag = False
    plot_parking_lots_flag = False
    plot_markers_flag = False
    plot_medians_flag = False
    plot_current_chargers_flag = False  # Stage 0
    plot_stage_1_flag = False
    plot_stage_2_flag = False
    plot_stage_3_flag = False
    plot_stage_4_flag = False
    plot_light_stations_flag = False
    plot_confirmed_chargers_flag = False  # not actually confirmed chargers but are lots suitable for EV chargers
    annotate_confirmed_chargers_flag = False
    plot_distances_flag = False

    # GRAPH INITIALIZATION
    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax = fig.add_subplot(111)

    # IMAGE AND GRAPH CONSTRAINTS
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

        if plot_neighborhoods_flag:
            ax.plot(x_n, y_n, color='gray')
            ax.fill(x_n, y_n, color='lightgray', alpha=0.5)
        coords_n = []

    # parking lots
    if plot_parking_lots_flag:
        plot_parking_lots_func(ax, df_pl)

    # current chargers
    if plot_current_chargers_flag:
        plot_current_chargers_func(ax, coords_c)

    # stages
    if plot_stage_1_flag:
        plot_current_chargers_func(ax, coords_c, s=10)
        plot_stage_1_func(ax, df_stage_1)

    if plot_stage_2_flag:
        plot_current_chargers_func(ax, coords_c, s=10)
        plot_stage_1_func(ax, df_stage_1, s=10)
        plot_stage_2_func(ax, df_stage_2)

    if plot_stage_3_flag:
        plot_current_chargers_func(ax, coords_c, s=10)
        plot_stage_1_func(ax, df_stage_1, s=10)
        plot_stage_2_func(ax, df_stage_2, s=10)
        plot_stage_3_func(ax, df_stage_3)

    if plot_stage_4_flag:
        plot_current_chargers_func(ax, coords_c, s=10)
        plot_stage_1_func(ax, df_stage_1, s=10)
        plot_stage_2_func(ax, df_stage_2, s=10)
        plot_stage_3_func(ax, df_stage_3, s=10)
        plot_stage_4_func(ax, df_stage_4)

    # confirmed chargers
    if plot_confirmed_chargers_flag:
        plot_confirmed_chargers_func(ax, df_confirmed_chargers, annotate_confirmed_chargers_flag)

    # distances to light stations
    if plot_distances_flag:
        plot_distances_func(ax, )

    # median coords
    if plot_medians_flag:
        x_m = [coord[0] for coord in coords_m]
        y_m = [coord[1] for coord in coords_m]
        ax.scatter(x_m, y_m, color='red', s=15, label="median")

    # light stations
    if plot_light_stations_flag:
        x_ls = [coord[0] for coord in coords_ls]
        y_ls = [coord[1] for coord in coords_ls]
        ax.scatter(x_ls, y_ls, color='yellow', s=15, label="light station")

    # boarder coords/markers for helping with image
    if plot_markers_flag:
        x_n = [coord[0] + .00 for coord in marker_coords]
        y_n = [coord[1] + .00 for coord in marker_coords]
        ax.scatter(x_n, y_n, color='yellow', s=8)

    ax.legend()

    # plt.xlabel("X axis")
    # plt.ylabel("Y axis")
    plt.title(plot_title)

    plt.grid(False)
    plt.show()


if __name__ == '__main__':
    main()
