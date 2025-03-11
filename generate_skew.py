import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import metpy.calc as mpcalc
from metpy.plots import Hodograph, SkewT
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
        "height_array": get_safe_values("geopotential_height"),
    }
    # print("Parsed Data:", parsed)
    return parsed


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
    skew.ax.set_xlabel(
        str.upper(f"Temperature ({temp.units:~P})"), weight="bold", fontsize=12
    )
    skew.ax.set_ylabel(
        str.upper(f"Pressure ({pressure.units:~P})"), weight="bold", fontsize=12
    )

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
