#!/usr/bin/env python

# Convert DAMIT shape model format (shape.txt) to Wavefront .obj format

import sys

if len(sys.argv) != 3:
    print("Usage " + str(sys.argv[0]) + "  <shape_file> <obj_file>")
    sys.exit(1)

shape_file = str.strip(sys.argv[1])

f_out = open(str.strip(sys.argv[2]), 'w')

with open(shape_file) as f_in:

    lines = f_in.readlines()

    nv,nf = map(int,lines[0].split())

    for i in range(nv):
        x,y,z = map(float,lines[1+i].split())
        f_out.write(f"v {x:0.6f} {y:0.6f} {z:0.6f}\n")

    for i in range(nf):
        x,y,z = map(int,lines[nv+i+1].split())
        f_out.write(f"f {x} {y} {z}\n")

    f_out.close()

# EOF
