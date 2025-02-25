# First let's start with some simple imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import metpy.calc as mpcalc
from metpy.cbook import get_test_data
from metpy.plots import add_metpy_logo, Hodograph, SkewT
from metpy.units import units

# Upper air data can easily be obtained using the siphon package, but for this example we will use some of MetPy’s sample data.
col_names = ['pressure', 'height', 'temperature', 'dewpoint', 'direction', 'speed']
df = pd.read_fwf(get_test_data('may4_sounding.txt', as_file_obj=False),
                 skiprows=5, usecols=[0, 1, 2, 3, 6, 7], names=col_names)

# Drop any rows with all NaN values for T, Td, winds
df = df.dropna(subset=('temperature', 'dewpoint', 'direction', 'speed'),
               how='all').reset_index(drop=True)

# We will pull the data out of the example dataset into individual variables and assign units.
p = df['pressure'].values * units.hPa
z = df['height'].values * units.m
T = df['temperature'].values * units.degC
Td = df['dewpoint'].values * units.degC
wind_speed = df['speed'].values * units.knots
wind_dir = df['direction'].values * units.degrees
u, v = mpcalc.wind_components(wind_speed, wind_dir)


# Now let’s make a Skew-T Log-P diagram using some simply MetPy functionality Create a new figure. The dimensions here give a good aspect ratio

fig = plt.figure(figsize=(9, 9))
add_metpy_logo(fig, 90, 80, size='small')
skew = SkewT(fig, rotation=45, rect=(0.1, 0.1, 0.55, 0.85))

# Plot the data using normal plotting functions, in this case using
# log scaling in Y, as dictated by the typical meteorological plot
skew.plot(p, T, 'r')
skew.plot(p, Td, 'g')
skew.plot_barbs(p, u, v)

# Change to adjust data limits and give it a semblance of what we want
skew.ax.set_adjustable('datalim')
# Vertical pressure levels 
skew.ax.set_ylim(1000, 100)
# Horizontal degrees in celcius 
skew.ax.set_xlim(-20, 30)

# Add the relevant special lines
skew.plot_dry_adiabats()
skew.plot_moist_adiabats()
skew.plot_mixing_lines()

# Create a hodograph
ax = plt.axes((0.7, 0.75, 0.2, 0.2))
h = Hodograph(ax, component_range=60.)
h.add_grid(increment=20)
h.plot(u, v)

plt.show()