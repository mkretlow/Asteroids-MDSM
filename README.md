# Asteroids-MDSM
I am sharing here a collection of tools and scripts for my own Asteroid Multi-Data (3D) Shape Modeling work.


### 3dshape_viewer.py
Displays simple 3D (asteroid, comets, etc.) shape files. Following packages are required: numpy and pyglet.

Three file formats are supported: Wavefront .obj file, DAMIT (and ADAM) shape files, Occult4 shape files (which are similar to DAMIT shape files, but comma seperated and with a blank line between the vertices and facet section).

```bash
Usage: 3dshape_viewer.py -o,-s,-t shape_file
-o: Wavefront .obj file
-s: DAMIT shape file
-t: Occult4 shape file (CSV)
```

The visualization does not consider the aspect angle from the Earth nor the pole orientation wrt to the Ecliptic. Possibly this feature will be added in the future.

Sample shape files are in the folder 'shapes'.

### shape2obj.py
Converts a DAMIT/ADAM shape file (shape.txt) to a Wavefront .obj file.

`Usage ./shape2obj.py  <shape_file> <obj_file>`



## Shape model files
- <https://astro.troja.mff.cuni.cz/projects/damit/>
- <https://sbn.psi.edu/pds/shape-models/>