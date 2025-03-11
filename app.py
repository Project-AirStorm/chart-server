import os
import io
import time
import requests
import boto3
from flask import Flask, request, jsonify
import io
import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import metpy.calc as mpcalc
from metpy.plots import Hodograph, SkewT
from metpy.units import units

from generate_skew import parse_json


app = Flask(__name__)

# Initialize S3 Stuff
s3_client = boto3.client("s3", region_name="us-east-1")
BUCKET_NAME = "meteo-charts"  # S3 bucket
FOLDER_NAME = "skewt-svg-dumps"  # S3 bucket folder


def plot_skewt(parsed_data, output_filename=None):
    """
    Given parsed raw data arrays, attach units and plot the SkewT and Hodograph.
    """
    pressure = np.array(parsed_data["pressure_values"]) * units.hPa
    temp = np.array(parsed_data["temp_array"]) * units.degC
    dew_pt = np.array(parsed_data["dew_pt_array"]) * units.degC
    height = np.array(parsed_data["height_array"]) * units.m
    wind_spd = np.array(parsed_data["wind_spd_array"]) * units("km/h")
    wind_dir = np.array(parsed_data["wind_dir_array"]) * units.deg
    wind_speed = wind_spd.to("knots")

    # Convert speed/direction to U/V
    u, v = mpcalc.wind_components(wind_speed, wind_dir)

    #############################################
    """SKEW T"""
    #############################################

    fig = plt.figure(figsize=(18, 12))
    skew = SkewT(fig, rotation=45, rect=(0.05, 0.05, 0.50, 0.90))

    skew.ax.set_adjustable("datalim")
    skew.ax.set_ylim(1000, 100)
    skew.ax.set_xlim(-20, 30)

    # Set some better labels than the default to increase readability
    skew.ax.set_xlabel(str.upper(f"Temperature ({temp.units:~P})"), weight="bold", fontsize=12)
    skew.ax.set_ylabel(str.upper(f"Pressure ({pressure.units:~P})"), weight="bold", fontsize=12)

    # Set the facecolor of the skew-t object and the figure to white
    fig.set_facecolor("#ffffff")
    skew.ax.set_facecolor("#ffffff")

    # Make a shaded isotherm pattern.
    x1 = np.linspace(-100, 40, 8)
    x2 = np.linspace(-90, 50, 8)
    y = [1100, 50]
    for i in range(0, 8):
        skew.shade_area(y=y, x1=x1[i], x2=x2[i], color="gray", alpha=0.02, zorder=1)

    # Plot the data using normal plotting functions, in this case using
    # log scaling in Y, as dictated by the typical meteorological plot
    skew.plot(pressure, temp, "r", lw=4, label="TEMPERATURE")
    skew.plot(pressure, dew_pt, "g", lw=4, label="DEWPOINT")

    # Resample the wind barbs for a cleaner output with increased readability.
    interval = np.logspace(2, 3, 40) * units.hPa
    idx = mpcalc.resample_nn_1d(pressure, interval)
    skew.plot_barbs(pressure=pressure[idx], u=u[idx], v=v[idx])

    # Provide basic adjustments to linewidth and alpha to increase readability
    # ADd a matplotlib axvline to highlight the 0-degree isotherm
    skew.ax.axvline(0 * units.degC, linestyle="--", color="blue", alpha=0.3)
    skew.plot_dry_adiabats(lw=1, alpha=0.3)
    skew.plot_moist_adiabats(lw=1, alpha=0.3)
    skew.plot_mixing_lines(lw=1, alpha=0.3)

    # Calculate LCL height and plot as a black dot. Because `p`'s first value is
    # ~1000 mb and its last value is ~250 mb, the `0` index is selected for
    # `pressure`, `temp`, and `dew_pt` to lift the parcel from the surface. If `pressure` was inverted,
    # i.e. start from a low value, 250 mb, to a high value, 1000 mb, the `-1` index
    # should be selected.
    lcl_pressure, lcl_temperature = mpcalc.lcl(pressure[0], temp[0], dew_pt[0])
    skew.plot(lcl_pressure, lcl_temperature, "ko", markerfacecolor="black")

    # Calculate full parcel profile and add to plot as black line
    prof = mpcalc.parcel_profile(pressure, temp[0], dew_pt[0]).to("degC")
    skew.plot(pressure, prof, "k", linewidth=2, label="SB PARCEL PATH")

    # Shade areas of CAPE and CIN
    skew.shade_cin(pressure, temp, prof, dew_pt, alpha=0.2, label="SBCIN")
    skew.shade_cape(pressure, temp, prof, alpha=0.2, label="SBCAPE")

    #############################################
    """HODOGRAPH"""
    #############################################

    # Create a hodograph object
    hodo_ax = plt.axes((0.48, 0.45, 0.5, 0.5))
    hodograph = Hodograph(hodo_ax, component_range=80.0)  # was 80

    # Add two separate grid increments for readability
    hodograph.add_grid(increment=20, ls="-", lw=1.5, alpha=0.5)
    hodograph.add_grid(increment=10, ls="--", lw=1, alpha=0.2)

    # Removing tick marks, tick labels, and axis labels for cleaner look
    hodograph.ax.set_box_aspect(1)
    hodograph.ax.set_yticklabels([])
    hodograph.ax.set_xticklabels([])
    hodograph.ax.set_xticks([])
    hodograph.ax.set_yticks([])
    hodograph.ax.set_xlabel(" ")
    hodograph.ax.set_ylabel(" ")

    # Adds tick marks to the inside of the hodograph plot to increase readability
    plt.xticks(np.arange(0, 0, 1))
    plt.yticks(np.arange(0, 0, 1))
    for i in range(10, 120, 10):
        hodograph.ax.annotate(
            str(i),
            (i, 0),
            xytext=(0, 2),
            textcoords="offset pixels",
            clip_on=True,
            fontsize=10,
            weight="bold",
            alpha=0.3,
            zorder=0,
        )
    for i in range(10, 120, 10):
        hodograph.ax.annotate(
            str(i),
            (0, i),
            xytext=(0, 2),
            textcoords="offset pixels",
            clip_on=True,
            fontsize=10,
            weight="bold",
            alpha=0.3,
            zorder=0,
        )

    # plot the hodograph itself, using plot_colormapped, colored
    # by height
    hodograph.plot_colormapped(u, v, c=height, linewidth=3, label="0-12km WIND")
    # compute Bunkers storm motion so we can plot it on the hodograph!
    RM, LM, MW = mpcalc.bunkers_storm_motion(pressure, u, v, height)
    hodograph.ax.text(
        (RM[0].m + 0.5),
        (RM[1].m - 0.5),
        "RM",
        weight="bold",
        ha="left",
        fontsize=13,
        alpha=0.6,
    )
    hodograph.ax.text(
        (LM[0].m + 0.5),
        (LM[1].m - 0.5),
        "LM",
        weight="bold",
        ha="left",
        fontsize=13,
        alpha=0.6,
    )
    hodograph.ax.text(
        (MW[0].m + 0.5),
        (MW[1].m - 0.5),
        "MW",
        weight="bold",
        ha="left",
        fontsize=13,
        alpha=0.6,
    )
    hodograph.ax.arrow(
        0,
        0,
        RM[0].m - 0.3,
        RM[1].m - 0.3,
        linewidth=2,
        color="black",
        alpha=0.2,
        label="Bunkers RM Vector",
        length_includes_head=True,
        head_width=2,
    )

    ##############################################
    """CONVECTIVE DIAGNOSTICS"""
    ##############################################

    # Add a simple rectangle using Matplotlib's 'patches'
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

    # Now let's take a moment to calculate some simple severe-weather parameters using
    # metpy's calculations
    # Here are some classic severe parameters!
    kindex = mpcalc.k_index(pressure, temp, dew_pt)
    total_totals = mpcalc.total_totals_index(pressure, temp, dew_pt)

    # mixed layer parcel properties!
    ml_t, ml_td = mpcalc.mixed_layer(pressure, temp, dew_pt, depth=50 * units.hPa)
    ml_p, _, _ = mpcalc.mixed_parcel(pressure, temp, dew_pt, depth=50 * units.hPa)
    mlcape, mlcin = mpcalc.mixed_layer_cape_cin(
        pressure, temp, prof, depth=50 * units.hPa
    )

    # most unstable parcel properties!
    mu_p, mu_t, mu_td, _ = mpcalc.most_unstable_parcel(
        pressure, temp, dew_pt, depth=50 * units.hPa
    )
    mucape, mucin = mpcalc.most_unstable_cape_cin(
        pressure, temp, dew_pt, depth=50 * units.hPa
    )

    # Estimate height of LCL in meters from hydrostatic thickness (for sig_tor)
    new_p = np.append(pressure[pressure > lcl_pressure], lcl_pressure)
    new_t = np.append(temp[pressure > lcl_pressure], lcl_temperature)
    lcl_height = mpcalc.thickness_hydrostatic(new_p, new_t)

    # Compute Surface-based CAPE
    sbcape, sbcin = mpcalc.surface_based_cape_cin(pressure, temp, dew_pt)
    # Compute SRH
    (u_storm, v_storm), *_ = mpcalc.bunkers_storm_motion(pressure, u, v, height)
    *_, total_helicity1 = mpcalc.storm_relative_helicity(
        height, u, v, depth=1 * units.km, storm_u=u_storm, storm_v=v_storm
    )
    *_, total_helicity3 = mpcalc.storm_relative_helicity(
        height, u, v, depth=3 * units.km, storm_u=u_storm, storm_v=v_storm
    )
    *_, total_helicity6 = mpcalc.storm_relative_helicity(
        height, u, v, depth=6 * units.km, storm_u=u_storm, storm_v=v_storm
    )

    # Copmute Bulk Shear components and then magnitude
    ubshr1, vbshr1 = mpcalc.bulk_shear(
        pressure, u, v, height=height, depth=1 * units.km
    )
    bshear1 = mpcalc.wind_speed(ubshr1, vbshr1)
    ubshr3, vbshr3 = mpcalc.bulk_shear(
        pressure, u, v, height=height, depth=3 * units.km
    )
    bshear3 = mpcalc.wind_speed(ubshr3, vbshr3)
    ubshr6, vbshr6 = mpcalc.bulk_shear(
        pressure, u, v, height=height, depth=6 * units.km
    )
    bshear6 = mpcalc.wind_speed(ubshr6, vbshr6)

    # Use all computed pieces to calculate the Significant Tornado parameter
    sig_tor = mpcalc.significant_tornado(
        sbcape, lcl_height, total_helicity3, bshear3
    ).to_base_units()

    # Perform the calculation of supercell composite if an effective layer exists
    super_comp = mpcalc.supercell_composite(mucape, total_helicity3, bshear3)

    # fig = plt.figure(figsize=(18, 12))
    # fig.set_facecolor("#ffffff")

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

    # plt.figtext(
    #     0.45,
    #     0.97,
    #     "Shreveport, LA VERTICAL PROFILE",
    #     weight="bold",
    #     fontsize=20,
    #     ha="center",
    # )
    ####################################################
    # Add legends to the skew and hodo
    skewleg = skew.ax.legend(loc="upper left")
    hodoleg = hodograph.ax.legend(loc="upper left")

    
    svg_buffer = io.BytesIO()
    plt.savefig(svg_buffer, format="svg", transparent=True)
    plt.close(fig)
    svg_buffer.seek(0)

    return svg_buffer.getvalue()











@app.route("/generate-skew", methods=["GET"])
def generate_skew():
    """
    Example endpoint: /generate-skew?days=3&lat=32.52&lon=-93.75
    """
    # Grab the query parameters
    forecast_days = request.args.get("days", type=int)
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)

    # Bomb out if a bad GET request was sent
    if forecast_days is None or lat is None or lon is None:
        return jsonify({"error": "Must provide 'days', 'lat', and 'lon'"}), 400


    # 2. Build the Open-Meteo API URL based on these params
    #    Here, we just replicate your example with the included fields:
    #    (Note: if you want fewer variables, you can strip them out.)
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}"
        f"&longitude={lon}"
        "&hourly="
        "temperature_1000hPa,temperature_975hPa,temperature_950hPa,temperature_925hPa,"
        "temperature_900hPa,temperature_875hPa,temperature_850hPa,temperature_825hPa,"
        "temperature_800hPa,temperature_775hPa,temperature_750hPa,temperature_725hPa,"
        "temperature_700hPa,temperature_675hPa,temperature_650hPa,temperature_625hPa,"
        "temperature_600hPa,temperature_575hPa,temperature_550hPa,temperature_525hPa,"
        "temperature_500hPa,temperature_475hPa,temperature_450hPa,temperature_425hPa,"
        "temperature_400hPa,temperature_375hPa,temperature_350hPa,temperature_325hPa,"
        "temperature_300hPa,temperature_275hPa,temperature_250hPa,temperature_225hPa,"
        "temperature_200hPa,temperature_175hPa,temperature_150hPa,temperature_125hPa,"
        "temperature_100hPa,temperature_70hPa,temperature_50hPa,temperature_40hPa,"
        "temperature_30hPa,temperature_20hPa,temperature_15hPa,temperature_10hPa,"
        "dew_point_1000hPa,dew_point_975hPa,dew_point_950hPa,dew_point_925hPa,"
        "dew_point_900hPa,dew_point_875hPa,dew_point_850hPa,dew_point_825hPa,"
        "dew_point_800hPa,dew_point_775hPa,dew_point_750hPa,dew_point_725hPa,"
        "dew_point_700hPa,dew_point_675hPa,dew_point_650hPa,dew_point_625hPa,"
        "dew_point_600hPa,dew_point_575hPa,dew_point_550hPa,dew_point_525hPa,"
        "dew_point_500hPa,dew_point_475hPa,dew_point_450hPa,dew_point_425hPa,"
        "dew_point_400hPa,dew_point_375hPa,dew_point_350hPa,dew_point_325hPa,"
        "dew_point_300hPa,dew_point_275hPa,dew_point_250hPa,dew_point_225hPa,"
        "dew_point_200hPa,dew_point_175hPa,dew_point_150hPa,dew_point_125hPa,"
        "dew_point_100hPa,dew_point_70hPa,dew_point_50hPa,dew_point_40hPa,"
        "dew_point_30hPa,dew_point_20hPa,dew_point_15hPa,dew_point_10hPa,"
        "wind_speed_1000hPa,wind_speed_975hPa,wind_speed_950hPa,wind_speed_925hPa,"
        "wind_speed_900hPa,wind_speed_875hPa,wind_speed_850hPa,wind_speed_825hPa,"
        "wind_speed_800hPa,wind_speed_775hPa,wind_speed_750hPa,wind_speed_725hPa,"
        "wind_speed_700hPa,wind_speed_675hPa,wind_speed_650hPa,wind_speed_625hPa,"
        "wind_speed_600hPa,wind_speed_575hPa,wind_speed_550hPa,wind_speed_525hPa,"
        "wind_speed_500hPa,wind_speed_475hPa,wind_speed_450hPa,wind_speed_425hPa,"
        "wind_speed_400hPa,wind_speed_375hPa,wind_speed_350hPa,wind_speed_325hPa,"
        "wind_speed_300hPa,wind_speed_275hPa,wind_speed_250hPa,wind_speed_225hPa,"
        "wind_speed_200hPa,wind_speed_175hPa,wind_speed_150hPa,wind_speed_125hPa,"
        "wind_speed_100hPa,wind_speed_70hPa,wind_speed_50hPa,wind_speed_40hPa,"
        "wind_speed_30hPa,wind_speed_20hPa,wind_speed_15hPa,wind_speed_10hPa,"
        "wind_direction_1000hPa,wind_direction_975hPa,wind_direction_950hPa,"
        "wind_direction_925hPa,wind_direction_900hPa,wind_direction_875hPa,"
        "wind_direction_850hPa,wind_direction_825hPa,wind_direction_800hPa,"
        "wind_direction_775hPa,wind_direction_750hPa,wind_direction_725hPa,"
        "wind_direction_700hPa,wind_direction_675hPa,wind_direction_650hPa,"
        "wind_direction_625hPa,wind_direction_600hPa,wind_direction_575hPa,"
        "wind_direction_550hPa,wind_direction_525hPa,wind_direction_500hPa,"
        "wind_direction_475hPa,wind_direction_450hPa,wind_direction_425hPa,"
        "wind_direction_400hPa,wind_direction_375hPa,wind_direction_350hPa,"
        "wind_direction_325hPa,wind_direction_300hPa,wind_direction_275hPa,"
        "wind_direction_250hPa,wind_direction_225hPa,wind_direction_200hPa,"
        "wind_direction_175hPa,wind_direction_150hPa,wind_direction_125hPa,"
        "wind_direction_100hPa,wind_direction_70hPa,wind_direction_50hPa,"
        "wind_direction_40hPa,wind_direction_30hPa,wind_direction_20hPa,"
        "wind_direction_15hPa,wind_direction_10hPa,"
        "geopotential_height_1000hPa,geopotential_height_975hPa,"
        "geopotential_height_950hPa,geopotential_height_925hPa,"
        "geopotential_height_900hPa,geopotential_height_875hPa,"
        "geopotential_height_850hPa,geopotential_height_825hPa,"
        "geopotential_height_800hPa,geopotential_height_775hPa,"
        "geopotential_height_750hPa,geopotential_height_725hPa,"
        "geopotential_height_700hPa,geopotential_height_675hPa,"
        "geopotential_height_650hPa,geopotential_height_625hPa,"
        "geopotential_height_600hPa,geopotential_height_575hPa,"
        "geopotential_height_550hPa,geopotential_height_525hPa,"
        "geopotential_height_500hPa,geopotential_height_475hPa,"
        "geopotential_height_450hPa,geopotential_height_425hPa,"
        "geopotential_height_400hPa,geopotential_height_375hPa,"
        "geopotential_height_350hPa,geopotential_height_325hPa,"
        "geopotential_height_300hPa,geopotential_height_275hPa,"
        "geopotential_height_250hPa,geopotential_height_225hPa,"
        "geopotential_height_200hPa,geopotential_height_175hPa,"
        "geopotential_height_150hPa,geopotential_height_125hPa,"
        "geopotential_height_100hPa,geopotential_height_70hPa,"
        "geopotential_height_50hPa,geopotential_height_40hPa,"
        "geopotential_height_30hPa,geopotential_height_20hPa,"
        "geopotential_height_15hPa,geopotential_height_10hPa"
        "&forecast_days=7"
        f"&forecast_days={forecast_days}"
        "&models=gfs_graphcast025"
    )
    # 3. Fetch the weather data from Open-Meteo
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        weather_json = response.json()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    hours_to_process = forecast_days * 24
    all_s3_urls = []

    # Iterate over hours and generate SVGs
    for hour in range(hours_to_process):
        parsed_data = parse_json(weather_json, hour_index=hour)

        # Generate the SkewT/Hodo as an in-memory SVG
        # in-memory BytesIO to avoid local filesystem usage
        out_filename = f"skewt_{lat}_{lon}_hour_{hour}.svg"

        # The plot_skewt_from_json can accept a file path or file-like.
        # We can save directly to the BytesIO, but we need a small tweak:
        svg_data = plot_skewt(parsed_data, output_filename=None)

        # Re-open that saved file from disk OR you can modify plot_skewt_from_json
        # to accept a file-like object. For simplicity, let's assume it currently
        # must save to disk, so let's read it back.
        # with open(out_filename, "rb") as f:
        #     svg_data = f.read()

        # Now upload the bytes to S3
        s3_key = f"{'skewt-svg-dumps'}/skewt_hour_{hour}.svg"
        try:
            s3_client.put_object(
                Bucket=BUCKET_NAME,
                Key=s3_key,
                Body=svg_data,
                ContentType="image/svg+xml",
            )
            # Construct a URL to the uploaded file (if the bucket is public or has a CloudFront distribution, etc.)
            # For a non-public bucket, you’ll have to generate presigned URLs or set the ACL accordingly.
            print(f"Uploaded {s3_key} to S3 successfully.")
            file_url = f"https://{BUCKET_NAME}.s3.amazonaws.com/{s3_key}"
            all_s3_urls.append(file_url)
        except Exception as e:
            print(f"Error uploading to S3: {e}")

    # Return JSON respone, describes where the files are stored
    return jsonify(
        {
            "latitude": lat,
            "longitude": lon,
            "days_requested": forecast_days,
            "s3_files": all_s3_urls,
            "message": "SkewT images generated and uploaded to S3.",
        }
    )


if __name__ == "__main__":
    # Make sure to allow inbound traffic on the port in your EC2 Security Group
    app.run(host="0.0.0.0", port=5000, debug=True)
