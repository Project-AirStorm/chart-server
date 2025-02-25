# import json
# import matplotlib.pyplot as plt
# import metpy.calc as mpcalc
# from metpy.units import units
# from metpy.plots import Skewt, Hodograph

# def view_json():
    
#     #print keys
#     with open('forecast.json', 'r') as file: 
#         weather_data = json.load(file)
#     for key in weather_data:
#         print(key)
    
#     hourly_data = weather_data['hourly']
#     for temps in hourly_data:
#         print(temps)

#     print(temperature_1000hpacv)
#     #print(json.dumps(weather_data.hourly, indent=4))

# view_json()

# # def parse_json(json_data, hour_index=0):    
    

import json

# First let's start with some simple imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import metpy.calc as mpcalc
from metpy.cbook import get_test_data
from metpy.plots import add_metpy_logo, Hodograph, SkewT
from metpy.units import units



parameters = ['temperature', 'dew_point', 'wind_speed', 'wind_direction', 'geopotential_height']

def parse_json(weather_json, hour_index):
    # Iterate over array, if it contains null is casts it a np.nan, which makes the processing below easier. 
    
    '''
    Returns the raw soundings at each hPa FOR THE HOUR passed in. 

    temperature_600hPa	
    0:	-12.7
    ...
    dew_point_1000hPa	
    0:	5.3

    '''

    #print(weather_array)
    hourly = weather_json['hourly']
    pressure_values = [1000, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100]

    def get_safe_values(weather_variable):
        values = []
        for press in pressure_values:
            key = f"{weather_variable}_{press}hPa"
            # Gets the list for given weather variable key ("temperature"), if its missings defaults to a list with np.nan 
            data_list = hourly.get(key, [np.nan])

            print(f"{key}: {data_list}")
            
            # If the list exists but is too short or the value is None, use np.nan.
            if len (data_list) > hour_index and data_list[hour_index] is not None: 
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
        "wind_dir_array": get_safe_values("wind_direction")
    }
    # print("Parsed Data:", parsed)  # Quick console log here
    return parsed



def plot_skewt_from_json(parsed_data):
    """
    Given parsed raw data arrays, attach units and plot the SkewT and Hodograph.
    """
    pressures = np.array(parsed_data["pressure_values"]) * units.hPa
    temp_array = np.array(parsed_data["temp_array"]) * units.degC 
    dew_point_array = np.array(parsed_data["dew_pt_array"]) * units.degC
    wind_spd_array = np.array(parsed_data["wind_spd_array"]) * units("km/h")
    wind_direction = np.array(parsed_data["wind_dir_array"]) * units.deg
    wind_speed = wind_spd_array.to("knots")

    # Convert speed/direction to U/V
    u, v = mpcalc.wind_components(wind_speed, wind_direction)



    # -- Now set up the SkewT figure
    fig = plt.figure(figsize=(9, 9))
    # add_metpy_logo(fig, 100, 80, size='small')
    
    skew = SkewT(fig, rotation=45, rect=(0.1, 0.1, 0.55, 0.85))

    # -- Plot data
    skew.plot(pressures, temp_array, 'r')
    skew.plot(pressures, dew_point_array, 'g')
    skew.plot_barbs(pressures, u, v)

    # Set axis limits
    skew.ax.set_ylim(1000, 100)  # or auto
    skew.ax.set_xlim(-40, 30)    # adjust as needed

    # Add standard lines (optional)
    skew.plot_dry_adiabats()
    skew.plot_moist_adiabats()
    skew.plot_mixing_lines()

    # -- Hodograph (optional)
    # Create a new axes for the hodograph
    ax_hod = plt.axes((0.7, 0.75, 0.2, 0.2))
    h = Hodograph(ax_hod, component_range=80.)  # max range in knots
    h.add_grid(increment=20)
    h.plot(u, v)  # plot the wind profile

    # Show or save
    plt.show()
    # Or: plt.savefig("skewt_example.png", dpi=150)




# Example usage:
if __name__ == "__main__":
    # Suppose 'raw_json_str' is your JSON data as a string
    # raw_json_str = <the big JSON you have>
    # Or read from file:
    # with open('my_sounding_data.json', 'r') as f:
    #     raw_json_str = f.read()

    # For demonstration, pretend we have the JSON in a variable:
    raw_json_str = """{ "hourly": { ... } }"""  # truncated for brevity

    # 1) Parse the JSON into a Python dict
    with open('forecast.json', 'r') as file:
        data_dict = json.load(file)


    # parse_json(data_dict, 0)
    parsed_data = parse_json(data_dict, hour_index=0)

    plot_skewt_from_json(parsed_data)

    
    # 2) Plot for hour_index = 0 (2025-02-25T00:00)
    
    
    #plot_skewt_from_json(data_dict, hour_index=0)
