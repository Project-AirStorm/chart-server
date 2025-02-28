import json
import os
import time
import matplotlib.pyplot as plt, mpld3
import numpy as np
import pandas as pd
import metpy.calc as mpcalc
from metpy.cbook import get_test_data
from metpy.plots import add_metpy_logo, Hodograph, SkewT
from metpy.units import units


parameters = [
    "temperature",
    "dew_point",
    "wind_speed",
    "wind_direction",
    "geopotential_height",
]


def parse_json(weather_json, hour_index):
    """
    Iterates over the JSON array, if it contains null is casts it a np.nan, which makes the processing below easier.

    Returns the raw soundings at each hPa FOR THE HOUR passed in.

    temperature_600hPa
    0:	-12.7
    ...
    dew_point_1000hPa
    0:	5.3

    """

    # print(weather_array)
    hourly = weather_json["hourly"]
    pressure_values = [1000, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100]

    def get_safe_values(weather_variable):
        values = []
        for press in pressure_values:
            key = f"{weather_variable}_{press}hPa"
            # Gets the list for given weather variable key ("temperature"), if its missings defaults to a list with np.nan
            data_list = hourly.get(key, [np.nan])

            # print(f"{key}: {data_list}")

            # If the list exists but is too short or the value is None, use np.nan.
            if len(data_list) > hour_index and data_list[hour_index] is not None:
                # If the value is a list with a single element, extract that element.
                val = data_list[hour_index]
                if isinstance(val, list) and len(val) == 1:
                    values.append(val[0])
                else:
                    values.append(val)
            else:
                values.append(np.nan)
        return values

    parsed = {
        "pressure_values": pressure_values,
        "temp_array": get_safe_values("temperature"),
        "dew_pt_array": get_safe_values("dew_point"),
        "wind_spd_array": get_safe_values("wind_speed"),
        "wind_dir_array": get_safe_values("wind_direction"),
        "height": get_safe_values("geopotential_height"),
    }
    # print("Parsed Data:", parsed)
    return parsed


def plot_skewt_from_json(parsed_data, output_filename=None):
    """
    Given parsed raw data arrays, attach units and plot the SkewT and Hodograph.
    """
    p = np.array(parsed_data["pressure_values"]) * units.hPa
    T = np.array(parsed_data["temp_array"]) * units.degC
    Td = np.array(parsed_data["dew_pt_array"]) * units.degC
    z = np.array(parsed_data["height"]) * units.m
    wind_spd_array = np.array(parsed_data["wind_spd_array"]) * units("km/h")
    wind_direction = np.array(parsed_data["wind_dir_array"]) * units.deg
    wind_speed = wind_spd_array.to("knots")

    # Convert speed/direction to U/V
    u, v = mpcalc.wind_components(wind_speed, wind_direction)

    lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], T[0], Td[0])
    prof = mpcalc.parcel_profile(p, T[0], Td[0]).to("degC")

    # Now let's take a moment to calculate some simple severe-weather parameters using
    # metpy's calculations
    # Here are some classic severe parameters!
    kindex = mpcalc.k_index(p, T, Td)
    total_totals = mpcalc.total_totals_index(p, T, Td)

    # mixed layer parcel properties!
    ml_t, ml_td = mpcalc.mixed_layer(p, T, Td, depth=50 * units.hPa)
    ml_p, _, _ = mpcalc.mixed_parcel(p, T, Td, depth=50 * units.hPa)
    mlcape, mlcin = mpcalc.mixed_layer_cape_cin(p, T, prof, depth=50 * units.hPa)

    # most unstable parcel properties!
    mu_p, mu_t, mu_td, _ = mpcalc.most_unstable_parcel(p, T, Td, depth=50 * units.hPa)
    mucape, mucin = mpcalc.most_unstable_cape_cin(p, T, Td, depth=50 * units.hPa)

    # Estimate height of LCL in meters from hydrostatic thickness (for sig_tor)
    new_p = np.append(p[p > lcl_pressure], lcl_pressure)
    new_t = np.append(T[p > lcl_pressure], lcl_temperature)
    lcl_height = mpcalc.thickness_hydrostatic(new_p, new_t)

    # Compute Surface-based CAPE
    sbcape, sbcin = mpcalc.surface_based_cape_cin(p, T, Td)
    # Compute SRH
    (u_storm, v_storm), *_ = mpcalc.bunkers_storm_motion(p, u, v, z)
    *_, total_helicity1 = mpcalc.storm_relative_helicity(
        z, u, v, depth=1 * units.km, storm_u=u_storm, storm_v=v_storm
    )
    *_, total_helicity3 = mpcalc.storm_relative_helicity(
        z, u, v, depth=3 * units.km, storm_u=u_storm, storm_v=v_storm
    )
    *_, total_helicity6 = mpcalc.storm_relative_helicity(
        z, u, v, depth=6 * units.km, storm_u=u_storm, storm_v=v_storm
    )

    # Copmute Bulk Shear components and then magnitude
    ubshr1, vbshr1 = mpcalc.bulk_shear(p, u, v, height=z, depth=1 * units.km)
    bshear1 = mpcalc.wind_speed(ubshr1, vbshr1)
    ubshr3, vbshr3 = mpcalc.bulk_shear(p, u, v, height=z, depth=3 * units.km)
    bshear3 = mpcalc.wind_speed(ubshr3, vbshr3)
    ubshr6, vbshr6 = mpcalc.bulk_shear(p, u, v, height=z, depth=6 * units.km)
    bshear6 = mpcalc.wind_speed(ubshr6, vbshr6)

    # Use all computed pieces to calculate the Significant Tornado parameter
    sig_tor = mpcalc.significant_tornado(
        sbcape, lcl_height, total_helicity3, bshear3
    ).to_base_units()

    # Perform the calculation of supercell composite if an effective layer exists
    super_comp = mpcalc.supercell_composite(mucape, total_helicity3, bshear3)

    fig = plt.figure(figsize=(18, 12))
    fig.set_facecolor("#ffffff")
    fig.patches.extend(
        [
            plt.Rectangle(
                (0.563, 0.05),
                0.334,
                0.37,
                edgecolor="black",
                facecolor="white",
                linewidth=1,
                alpha=1,
                transform=fig.transFigure,
                figure=fig,
            )
        ]
    )

    # There is a lot we can do with this data operationally, so let's plot some of
    # these values right on the plot, in the box we made
    # First lets plot some thermodynamic parameters
    plt.figtext(
        0.58, 0.37, "SBCAPE: ", weight="bold", fontsize=15, color="black", ha="left"
    )
    plt.figtext(
        0.71,
        0.37,
        f"{sbcape:.0f~P}",
        weight="bold",
        fontsize=15,
        color="orangered",
        ha="right",
    )
    plt.figtext(
        0.58, 0.34, "SBCIN: ", weight="bold", fontsize=15, color="black", ha="left"
    )
    plt.figtext(
        0.71,
        0.34,
        f"{sbcin:.0f~P}",
        weight="bold",
        fontsize=15,
        color="lightblue",
        ha="right",
    )
    plt.figtext(
        0.58, 0.29, "MLCAPE: ", weight="bold", fontsize=15, color="black", ha="left"
    )
    plt.figtext(
        0.71,
        0.29,
        f"{mlcape:.0f~P}",
        weight="bold",
        fontsize=15,
        color="orangered",
        ha="right",
    )
    plt.figtext(
        0.58, 0.26, "MLCIN: ", weight="bold", fontsize=15, color="black", ha="left"
    )
    plt.figtext(
        0.71,
        0.26,
        f"{mlcin:.0f~P}",
        weight="bold",
        fontsize=15,
        color="lightblue",
        ha="right",
    )
    plt.figtext(
        0.58, 0.21, "MUCAPE: ", weight="bold", fontsize=15, color="black", ha="left"
    )
    plt.figtext(
        0.71,
        0.21,
        f"{mucape:.0f~P}",
        weight="bold",
        fontsize=15,
        color="orangered",
        ha="right",
    )
    plt.figtext(
        0.58, 0.18, "MUCIN: ", weight="bold", fontsize=15, color="black", ha="left"
    )
    plt.figtext(
        0.71,
        0.18,
        f"{mucin:.0f~P}",
        weight="bold",
        fontsize=15,
        color="lightblue",
        ha="right",
    )
    plt.figtext(
        0.58, 0.13, "TT-INDEX: ", weight="bold", fontsize=15, color="black", ha="left"
    )
    plt.figtext(
        0.71,
        0.13,
        f"{total_totals:.0f~P}",
        weight="bold",
        fontsize=15,
        color="orangered",
        ha="right",
    )
    plt.figtext(
        0.58, 0.10, "K-INDEX: ", weight="bold", fontsize=15, color="black", ha="left"
    )
    plt.figtext(
        0.71,
        0.10,
        f"{kindex:.0f~P}",
        weight="bold",
        fontsize=15,
        color="orangered",
        ha="right",
    )

    # now some kinematic parameters
    plt.figtext(
        0.73, 0.37, "0-1km SRH: ", weight="bold", fontsize=15, color="black", ha="left"
    )
    plt.figtext(
        0.88,
        0.37,
        f"{total_helicity1:.0f~P}",
        weight="bold",
        fontsize=15,
        color="navy",
        ha="right",
    )
    plt.figtext(
        0.73,
        0.34,
        "0-1km SHEAR: ",
        weight="bold",
        fontsize=15,
        color="black",
        ha="left",
    )
    plt.figtext(
        0.88,
        0.34,
        f"{bshear1:.0f~P}",
        weight="bold",
        fontsize=15,
        color="blue",
        ha="right",
    )
    plt.figtext(
        0.73, 0.29, "0-3km SRH: ", weight="bold", fontsize=15, color="black", ha="left"
    )
    plt.figtext(
        0.88,
        0.29,
        f"{total_helicity3:.0f~P}",
        weight="bold",
        fontsize=15,
        color="navy",
        ha="right",
    )
    plt.figtext(
        0.73,
        0.26,
        "0-3km SHEAR: ",
        weight="bold",
        fontsize=15,
        color="black",
        ha="left",
    )
    plt.figtext(
        0.88,
        0.26,
        f"{bshear3:.0f~P}",
        weight="bold",
        fontsize=15,
        color="blue",
        ha="right",
    )
    plt.figtext(
        0.73, 0.21, "0-6km SRH: ", weight="bold", fontsize=15, color="black", ha="left"
    )
    plt.figtext(
        0.88,
        0.21,
        f"{total_helicity6:.0f~P}",
        weight="bold",
        fontsize=15,
        color="navy",
        ha="right",
    )
    plt.figtext(
        0.73,
        0.18,
        "0-6km SHEAR: ",
        weight="bold",
        fontsize=15,
        color="black",
        ha="left",
    )
    plt.figtext(
        0.88,
        0.18,
        f"{bshear6:.0f~P}",
        weight="bold",
        fontsize=15,
        color="blue",
        ha="right",
    )
    plt.figtext(
        0.73,
        0.13,
        "SIG TORNADO: ",
        weight="bold",
        fontsize=15,
        color="black",
        ha="left",
    )
    plt.figtext(
        0.88,
        0.13,
        f"{sig_tor[0]:.0f~P}",
        weight="bold",
        fontsize=15,
        color="orangered",
        ha="right",
    )
    plt.figtext(
        0.73,
        0.10,
        "SUPERCELL COMP: ",
        weight="bold",
        fontsize=15,
        color="black",
        ha="left",
    )
    plt.figtext(
        0.88,
        0.10,
        f"{super_comp[0]:.0f~P}",
        weight="bold",
        fontsize=15,
        color="orangered",
        ha="right",
    )

    plt.figtext(
        0.45,
        0.97,
        "Shreveport, LA VERTICAL PROFILE",
        weight="bold",
        fontsize=20,
        ha="center",
    )
        
    plt.savefig(output_filename, format="svg", transparent=True)
    plt.close(fig)

    # Show or save
    # plt.show()
    # Or: plt.savefig("skewt_example.png", dpi=150)



if __name__ == "__main__":

    # with open("data/forecast-okc-7-days.json", "r") as file:
    #     JSON_sounding = json.load(file)

    # parsed_data = parse_json(JSON_sounding, hour_index=0)
    # plot_skewt_from_json(parsed_data, output_filename=None)

    start_time = time.time()

    output_dir = "svg-dump"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open("data/forecast-shreveport-7-day.json", "r") as file:
        JSON_sounding = json.load(file)

    for hour in range(167):
        parsed_data = parse_json(JSON_sounding, hour_index=hour)
        out_file = os.path.join(output_dir, f"skewt_hour_{hour}.svg")
        print(f"Generating: {out_file}")
        # plot_skewt_from_json(parsed_data)
        plot_skewt_from_json(parsed_data, output_filename=out_file)
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.2f} seconds")
