#!/usr/bin/env python

# ------------------------------------------------------------------------------------------------------------------------------------
# Name       : 3dshape_viewer.py
# Purpose    : Asteroid 3D shape model viewer (DAMIT, ADAM etc.)
# Author     : Mike Kretlow (mike@kretlow.de), Licence: MIT
# Last change: 2020-10-03, 2019-11-01, 2012-06-25, 2012-06-06
# ------------------------------------------------------------------------------------------------------------------------------------

"""
Displays a 3d shape object file (wavefront format .obj), especially for use with asteroid shape modells from DAMIT, ADAM etc.

"""

import os
import sys
import pyglet
import numpy as np

from pyglet.gl import *
from pyglet.window import key
from math import pi, sin, cos
from euclid import *

version = '0.2'

helptxt = """
<font size=+3 color=#FF3030>
<b>3D shape model viewer</b>
</font><br/>
<font size=+0 color=#FFFFFF>
<b>Version {:s}</b>
</font>
<br/><br/>
<font size=+2 color=#00FF60>
I J K L = Move Light 0<br/>
Arrows  = Rotate Object<br/>
R       = Reset<br/>
A       = Accelerate<br/>
D       = Decelerate<br/>
S       = Take Screenshot<br/>
C       = Start/Stop Capture<br/>
<!--X       = Show/Hide Axes<br/-->
W       = Show/Hide Wireframe<br/>
T       = Tumble Object<br/>
F       = Show/Hide FPS<br/>
Space   = Start/Stop rotation<br/>
H       = This Help<br/>
Q, Esc  = Quit<br/>
</font>

""".format(version)


# Try and create a pyglet window with multisampling (antialiasing)
try:
    config = Config(sample_buffers=1, samples=4, depth_size=16, double_buffer=True,)
    window = pyglet.window.Window(resizable=True, config=config, vsync=False, visible=False) # "vsync=False" to check the framerate
except pyglet.window.NoSuchConfigException:
    # Fall back to no multisampling for old hardware
    window = pyglet.window.Window(resizable=True, visible=False)

label1 = pyglet.text.HTMLLabel(helptxt, # location=location,
                              width=0.70*window.width,
                              multiline=True, anchor_x='center', anchor_y='center')

fps_display = pyglet.window.FPSDisplay(window=window)   # https://pyglet.readthedocs.io/en/stable/programming_guide/time.html#displaying-the-frame-rate

#window.clear()



@window.event
def on_resize(width, height):

    if height==0: height=1
    # Keep text vertically centered in the window
    label1.y = window.height // 2.5

    # Override the default on_resize handler to create a 3D projection
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60., width / float(height), .1, 1000.)
    glMatrixMode(GL_MODELVIEW)

    return pyglet.event.EVENT_HANDLED


def update(dt):
    global autorotate
    global rot
    global icap

    if autorotate:
        if tumble:
            rot += Vector3(10, 20, 15) * dt * rotspeed
        else:
            rot += Vector3(0, 10, 0) * dt * rotspeed

        rot.x %= 360
        rot.y %= 360
        rot.z %= 360

    if capture:
        filename = os.path.join(capture_dir, "image_" + "{0:06d}".format(icap) + ".png")
        pyglet.image.get_buffer_manager().get_color_buffer().save(filename)
        icap += 1


def dismiss_dialog(dt):
    global showdialog
    showdialog = False


# Define a simple function to create ctypes arrays of floats:
def vec(*args):
    return (GLfloat * len(args))(*args)


@window.event
def on_draw():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glLoadIdentity()

    glTranslatef(0.0, 0.0, -3.5);
    glRotatef(rot.x, 0, 0, 1)
    glRotatef(rot.y, 0, 1, 0)
    glRotatef(rot.z, 1, 0, 0)

    if wireframe:
        glPolygonMode(GL_FRONT, GL_LINE)
    else:
        glPolygonMode(GL_FRONT, GL_FILL)

    if showaxes: add_axes()

    batch1.draw()

    if wireframe:
        glPolygonMode(GL_FRONT, GL_FILL)

    glActiveTexture(GL_TEXTURE0)
    glEnable(GL_TEXTURE_2D)
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)

    if showdialog:
        glLoadIdentity()
        glTranslatef(0, -200, -450)
        label1.draw()

    glLoadIdentity()
    glTranslatef(250, -290, -500)
    if showfps: fps_display.draw()

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glDisable(GL_TEXTURE_2D)


@window.event
def on_key_press(symbol, modifiers):
    global rot
    global iimg
    global autorotate
    global rotspeed
    global capture
    global wireframe
    global light0pos
    global light1pos
    global showdialog
    global showfps
    global showaxes
    global tumble

    if symbol == key.R:
        print('Reset')
        rot = Vector3(0, 0, 90)
        rotspeed = 2.0
    elif symbol == key.ESCAPE or symbol == key.Q: # ESC would do it anyway, but not "Q"
        pyglet.app.exit()
        return pyglet.event.EVENT_HANDLED
    elif symbol == key.H:
        showdialog = not showdialog
    elif symbol == key.SPACE:
        print('Rotate: ', not autorotate)
        autorotate = not autorotate
    elif symbol == key.A:
        print('Accelerate autorot')
        rotspeed = rotspeed + 0.15*rotspeed
    elif symbol == key.D:
        print('Decelerate autorot')
        rotspeed = rotspeed - 0.15*rotspeed
    elif symbol == key.X:
        showaxes = not showaxes
        print('Show Axes: ', showaxes)
    elif symbol == key.W:
        wireframe = not wireframe
        print('Show Wireframe: ', wireframe)
    elif symbol == key.F:
        showfps = not showfps
        print('Show FPS: ', showfps)
    elif symbol == key.T:
        tumble = not tumble
        print('Tumble On: ', tumble)
    elif symbol == key.S:
        filename = os.path.join(sshot_dir, "image_" + "{0:04d}".format(iimg) + ".png")
        pyglet.image.get_buffer_manager().get_color_buffer().save(filename)
        iimg += 1
        print("Screenshot saved to ", filename)
    elif symbol == key.C:
        capture = not capture
        if capture:
            print("Start capturing...")
        else:
            print("Stop capturing")
    elif symbol == key.LEFT:
        print('Stop + left')
        if autorotate:
            autorotate = False
        else:
            rot.y += -rotstep
            rot.y %= 360
    elif symbol == key.DOWN:
        print('Stop + down')
        if autorotate:
            autorotate = False
        else:
            rot.z += rotstep
            rot.z %= 360
    elif symbol == key.UP:
        print('Stop + up')
        if autorotate:
            autorotate = False
        else:
            rot.z += -rotstep
            rot.z %= 360
    elif symbol == key.RIGHT:
        print('Stop + right')
        if autorotate:
            autorotate = False
        else:
            rot.y += rotstep
            rot.y %= 360
    elif symbol == key.J:
        print('Light0 rotate left')
        tmp = light0pos[0]
        light0pos[0] = tmp * cos( lightstep ) - light0pos[2] * sin( lightstep )
        light0pos[2] = light0pos[2] * cos( lightstep ) + tmp * sin( lightstep )
        glLoadIdentity()
        glLightfv(GL_LIGHT0, GL_POSITION, vec(*light0pos))
    elif symbol == key.L:
        print('Light0 rotate right')
        tmp = light0pos[0]
        light0pos[0] = tmp * cos( -lightstep ) - light0pos[2] * sin( -lightstep )
        light0pos[2] = light0pos[2] * cos( -lightstep ) + tmp * sin( -lightstep )
        glLoadIdentity()
        glLightfv(GL_LIGHT0, GL_POSITION, vec(*light0pos))
    elif symbol == key.I:
        print('Light0 up')
        tmp = light0pos[1]
        light0pos[1] = tmp * cos( -lightstep ) - light0pos[2] * sin( -lightstep )
        light0pos[2] = light0pos[2] * cos( -lightstep ) + tmp * sin( -lightstep )
        glLoadIdentity()
        glLightfv(GL_LIGHT0, GL_POSITION, vec(*light0pos))
    elif symbol == key.K:
        print('Light0 down')
        tmp = light0pos[1]
        light0pos[1] = tmp * cos( lightstep ) - light0pos[2] * sin( lightstep )
        light0pos[2] = light0pos[2] * cos( lightstep ) + tmp * sin( lightstep )
        glLoadIdentity()
        glLightfv(GL_LIGHT0, GL_POSITION, vec(*light0pos))
    else:
        print('OTHER KEY')


# One-time GL setup
def scene_setup():

    global light0pos
    global light1pos
    global wireframe

    setup = [True, False, False]

    light0pos = [ -10.0,  -10.0, -20.0, 1.0]    # Positional light
    light1pos = [-20.0, -20.0, 20.0, 0.0]       # Infinitely away light

    #glClearColor(1, 1, 1, 1)
    #glColor4f(1.0, 0.0, 0.0, 0.5 )
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Simple light setup.  On Windows GL_LIGHT0 is enabled by default,
    # but this is not the case on Linux or Mac, so remember to always include it.
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHT1)

    # Some sample initial setups
    if setup[0]:
        glLightfv (GL_LIGHT0, GL_DIFFUSE, vec(1.0, 1.0, 1.0, 1.0))
        glLightfv (GL_LIGHT0, GL_AMBIENT, vec(0.3, 0.3, 0.3, 1.0))
        glLightfv (GL_LIGHT0, GL_SPECULAR,vec(0.0, 0.0, 0.0, 0.0))
        glLightfv (GL_LIGHT0, GL_POSITION,vec(2.0, 2.0, 2.0, 0.0))

    elif setup[1]:
        glLightfv(GL_LIGHT0, GL_POSITION, vec(*light0pos))
        glLightfv(GL_LIGHT0, GL_AMBIENT, vec(0.3, 0.3, 0.3, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, vec(0.9, 0.9, 0.9, 1.0))
        glLightfv(GL_LIGHT0, GL_SPECULAR, vec(1.0, 1.0, 1.0, 1.0))
        glLightfv(GL_LIGHT1, GL_POSITION, vec(*light1pos))
        glLightfv(GL_LIGHT1, GL_DIFFUSE, vec(.6, .6, .6, 1.0))
        glLightfv(GL_LIGHT1, GL_SPECULAR, vec(1.0, 1.0, 1.0, 1.0))

    if setup[2]:
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, vec(0.8, 0.5, 0.5, 1.0))
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, vec(0.3, 0.3, 0.3, 0.75))
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, vec(1, 1, 1, 1))
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 50)


# Show axes (TODO, not working properly)
def add_axes():
    global batch1
    #glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    #glLoadIdentity()
    #pyglet.graphics.draw(1,GL_LINES,('v3f',(0.,0.,0.)),('v3f',( 200.,0.,0.)),('c3B',(255,0,0)))
    batch1.add(6,GL_LINES,None,('v3f',(0.,0.,0., 2.,0.,0., 0.,0.,0., 0.,2.,0., 0.,0.,0., 0.,0.,2.)),('c3B',(255,255,255,255,255,255,255,255,255,255,255,255,255,0,0,255,0,0)))
    #    , 0.,0.,0., 0.,2.,0., 0.,0.,0., 0.,0.,2.)))


# Read wavefront .obj file
def load_obj(filename):

    data = np.genfromtxt(filename, dtype=[('type', np.dtype('U1')), ('points', np.float32, 3)], comments = '#')

    # Get vertices and faces
    vertices = data['points'][data['type'] == 'v']
    faces = (data['points'][data['type'] == 'f']-1).astype(np.uint32)

    # Build normals
    normals = build_normals(vertices,faces)

    # Scale vertices such that object is contained in [-1:+1,-1:+1,-1:+1]
    vmin, vmax =  vertices.min(), vertices.max()
    vertices = (2*(vertices-vmin)/(vmax-vmin) - 1)

    vertices = list(vertices.flatten())
    normals  = list(normals.flatten())
    indices  = list(faces.flatten())

    if DEBUG:
        print(vertices[:6])
        print(normals[:6])
        print(indices[:6])

    return vertices, normals, indices


# Load DAMIT shape file
def load_shp(filename):

    f = open(filename,'r')
    s = f.readline()

    nvert = int(s.split()[0])
    nfac = int(s.split()[1])

    vlist = np.zeros((nvert,3))
    tlist = np.zeros((nfac,3),dtype=np.int)

    # Read vertex list
    for i in range(nvert):
        s = f.readline()
        svert = s.split()
        vlist[i,] = np.asarray([float(svert[0]),float(svert[1]),float(svert[2])])

    for i in range(nfac):
        s  = f.readline()
        svert = s.split()
        tlist[i,] = np.asarray([int(svert[0]),int(svert[1]),int(svert[2])],dtype=np.int)

    f.close()

    vertices = np.asarray(vlist)
    faces = np.asarray(tlist-1)

    # Build normals
    normals = build_normals(vertices,faces)

    # Scale vertices such that object is contained in [-1:+1,-1:+1,-1:+1]
    vmin, vmax =  vertices.min(), vertices.max()
    vertices = (2*(vertices-vmin)/(vmax-vmin) - 1) #/ 1.5  # 1.5 = scale factor

    vertices = list(vertices.flatten())
    normals  = list(normals.flatten())
    indices  = list(faces.flatten())

    return vertices, normals, indices


# Read shape file from Occult (different format to DAMIT shape files)
def load_shp_occult(filename):

    sep = ','

    f = open(filename,'r')
    s = f.readline()

    nvert = int(s.split(sep)[0])
    nfac = int(s.split(sep)[1])

    vlist = np.zeros((nvert,3))
    tlist = np.zeros((nfac,3),dtype=np.int)

    # Read vertex list
    for i in range(nvert):
        s = f.readline()
        svert = [x.strip() for x in s.split(sep,2)]
        vlist[i,] = np.asarray([float(svert[0]),float(svert[1]),float(svert[2])])

    # Occult: skip blank line
    line = f.readline()

    for i in range(nfac):
        s = f.readline()
        svert = [x.strip() for x in s.split(sep,2)]

        # Occult counts index 0,1,2 instead of 1,2,3
        tlist[i,] = np.asarray([int(svert[0])+1,int(svert[1])+1,int(svert[2])+1],dtype=np.int)

    f.close()

    vertices = np.asarray(vlist)
    faces = np.asarray(tlist-1)

    # Build normals
    normals = build_normals(vertices,faces)

    # Scale vertices such that object is contained in [-1:+1,-1:+1,-1:+1]
    vmin, vmax =  vertices.min(), vertices.max()
    vertices = (2*(vertices-vmin)/(vmax-vmin) - 1) #/ 1.5  # 1.5 = scale factor

    vertices = list(vertices.flatten())
    normals  = list(normals.flatten())
    indices  = list(faces.flatten())

    return vertices, normals, indices


def build_normals(vertices,faces):

    T = vertices[faces]
    N = np.cross(T[::,1 ]-T[::,0], T[::,2]-T[::,0])
    L = np.sqrt(N[:,0]**2+N[:,1]**2+N[:,2]**2)
    N /= L[:, np.newaxis]
    normals = np.zeros(vertices.shape)
    normals[faces[:,0]] += N
    normals[faces[:,1]] += N
    normals[faces[:,2]] += N
    L = np.sqrt(normals[:,0]**2+normals[:,1]**2+normals[:,2]**2)
    normals /= L[:, np.newaxis]

    return normals


# -----------------------------------------------------------------------------
# Script starts here
# -----------------------------------------------------------------------------
pyglet.clock.schedule(update)
pyglet.clock.schedule_once(dismiss_dialog, 7.0)

win_title   = "3D shape model viewer"
capture_dir = "capture"
sshot_dir   = "screenshot"
#rot        = Vector3(45,45,90)
rot         = Vector3(0, 0, 90)
autorotate  = True
rotspeed    = 2.0
rotstep     = 10
lightstep   = 10 * pi/180
showaxes    = False
wireframe   = False
showdialog  = True
showfps     = False
capture     = False
tumble      = False
iimg        = 0
icap        = 0
DEBUG       = False


# Command line arguments
if len(sys.argv) != 3 or sys.argv[1].strip() == '-h':
    print("Usage " + str(sys.argv[0]) + " -o,-s,-t shape_file")
    print("-o: Wavefront .obj file")
    print("-s: DAMIT shape file")
    print("-t: Occult4 shape file (CSV)")
    sys.exit(1)

# Check for Screenshot (single images) and Capture directories and create them if not exist
if not os.path.exists(capture_dir):
    os.makedirs(capture_dir)

if not os.path.exists(sshot_dir):
    os.makedirs(sshot_dir)

ftype = sys.argv[1].strip()
filename = sys.argv[2].strip()


if ftype not in ['-o', '-s', '-t']:
    print(f"Unknown file type option : {ftype}")
    sys.exit(1)


# Read model file according to type of shape file
if ftype == '-o':
    vertices,normals,indices = load_obj(filename)

elif ftype == '-s':
    vertices,normals,indices = load_shp(filename)

elif ftype == '-t':
    vertices,normals,indices = load_shp_occult(filename)


color = len(vertices)//3*(0,0,0)

window.set_caption(win_title + ' : ' + os.path.basename(filename))
window.set_visible(True)

scene_setup()

batch1  = pyglet.graphics.Batch()
batch1.add_indexed( len(vertices)//3, GL_TRIANGLES, None, indices, ('v3f/static', vertices), ('c3B', color), ('n3f/static', normals) )

pyglet.app.run()

# EOF
