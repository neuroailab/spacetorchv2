"""
Definition of custom colormaps.
This file is mostly a recording of the colors from Nauhaus et al., 2012 used to
indicate different orientations and spatial frequencies; I like them slightly better
than built in HSV.
"""

import matplotlib as mpl
import numpy as np

# colormaps for Nauhaus data
nauhaus_raw_colormaps = {}
nauhaus_raw_colormaps["angles"] = np.array(
    [
        np.array((190, 60, 60, 255)) / 255.0,
        np.array((210, 160, 60, 255)) / 255.0,
        np.array((230, 230, 70, 255)) / 255.0,
        np.array((85, 150, 85, 255)) / 255.0,
        np.array((90, 165, 185, 255)) / 255.0,
        np.array((70, 80, 140, 255)) / 255.0,
        np.array((170, 85, 140, 255)) / 255.0,
        np.array((190, 60, 60, 255)) / 255.0,

        # np.array([1.0, 0.0, 0.0, 1.0]),
        # np.array([1.0, 0.741177211765447, 0.0, 1.0]),
        # np.array([0.5176455764691059, 1.0, 0.0, 1.0]),
        # np.array([0.0, 1.0, 0.22353062080241548, 1.0]),
        # np.array([0.0, 1.0, 0.9647031631761764, 1.0]),
        # np.array([0.0, 0.29411984742867114, 1.0, 1.0]),
        # np.array([0.44705736433677606, 0.0, 1.0, 1.0]),
        # np.array([1.0, 0.0, 0.8117654238977766, 1.0]),
        # np.array([1.0, 0.0, 0.0, 1.0]),
    ]
)
nauhaus_raw_colormaps["sfs"] = np.array(
    [
        np.array((80, 140, 185, 255)) / 255.0,
        np.array((90, 155, 190, 255)) / 255.0,
        np.array((110, 170, 106, 255)) / 255.0,
        np.array((160, 200, 85, 255)) / 255.0,
        np.array((220, 220, 70, 255)) / 255.0,
        np.array((210, 150, 60, 255)) / 255.0,
        np.array((200, 100, 50, 255)) / 255.0,
        np.array((190, 70, 50, 255)) / 255.0,

        # np.array([16,0,255,255]) / 255.0,
        # np.array([12,115,250,255]) / 255.0,
        # np.array([2,250,250,255]) / 255.0,
        # np.array([2,255,0,255]) / 255.0,
        # np.array([254,253,2,255]) / 255.0,
        # np.array([254,150,0,255]) / 255.0,
        # np.array([255,2,0,255]) / 255.0,
    ]
)
nauhaus_raw_colormaps["colors"] = np.array([(0.8, 0.8, 0.8, 1), (0.1, 0.1, 0.1, 1)])

nauhaus_colormaps = {}
for colormap_name, raw_colormap in nauhaus_raw_colormaps.items():
    cmap = mpl.colors.LinearSegmentedColormap.from_list(colormap_name, raw_colormap)
    nauhaus_colormaps[colormap_name] = cmap
