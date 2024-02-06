import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd
import re


def main():
    print("Hello, World!")
    coords_pl = []
    coords_m = []
    neighborhoods_names = []

    df_pl = pd.read_csv("parking_lots.csv")
    df_m = pd.read_csv("Median Coordinates.csv")
    df_n = pd.read_csv("neighborhoods.csv")

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

    for index, row in df_pl.iterrows():
        x_coord = row["X"]
        y_coord = row["Y"]
        coords_pl.append((x_coord, y_coord))

    for index, row in df_m.iterrows():
        x_coord = row["X / Weights"]
        y_coord = row["Y / Weights"]
        coords_m.append((x_coord, y_coord))

    graph_plot(coords_pl, wkt_n_dict, neighborhoods_names, coords_m)


def graph_plot(coords_pl, wkt_n_dict, neighborhoods_names, coords_m) -> None:
    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax1 = fig.add_subplot(111)

    coords_n = []
    for name in neighborhoods_names:
        for coords in wkt_n_dict[name]:
            coords_n.append(coords)
        x_n = [coord[0] for coord in coords_n]
        y_n = [coord[1] for coord in coords_n]

        ax1.plot(x_n, y_n, color='gray')
        ax1.fill(x_n, y_n, color='lightblue', alpha=0.5)
        coords_n = []

    x_pl = [coord[0] for coord in coords_pl]
    y_pl = [coord[1] for coord in coords_pl]

    x_m = [coord[0] for coord in coords_m]
    y_m = [coord[1] for coord in coords_m]

    for coords in coords_m:
        center = (coords[0], coords[1])
        meters = 200.0


        for i in range(0, 1):
            radius = meters / 111_111.11
            print(radius)
            circle = Circle(center, radius, color='red', fill=False)
            ax1.add_artist(circle)
            meters += 200

    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.title("Parking Lots and Chargers")

    ax1.scatter(x_pl, y_pl, s=30)
    ax1.scatter(x_m, y_m, s=30, color='red')
    # plt.legend()
    plt.grid(False)
    plt.show()


if __name__ == '__main__':
    main()
