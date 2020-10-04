# Asteroids-MDSM
A collection of some tools and scripts formy own Asteroid Multi-Data (3D) Shape Modeling work.


### 3dshape_viewer.py
Displays simple shape files. Following packages are required: numpy and pyglet.

Three file formats are supported: Wafevefront .obj file, DAMIT (and ADAM) shape files, Occult4 shape files (similar to DAMIT shape files but comma seperated and blank line between vertices and facet section).

```bash
Usage: 3dshape_viewer.py -o,-s,-t shape_file
-o: Wavefront .obj file
-s: DAMIT shape file
-t: Occult4 shape file (CSV)
```


### shape2obj.py
Converts a DAMIT,ADAM shape file (shape.txt) to a Wavefront .obj file.

`Usage ./shape2obj.py  <shape_file> <obj_file>`

