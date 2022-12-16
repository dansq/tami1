from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

from bezier import BezierCurve, BezierSurface
import numpy as np

w, h = 500,500

# ---Section 1---

def draw_line(points):
    glBegin(GL_LINE_STRIP)
    for point in points:
        glVertex2f(*point)
    glEnd()

def square():
    # We have to declare the points in this sequence: bottom left, bottom right, top right, top left
    glBegin(GL_QUADS) # Begin the sketch
    glVertex2f(100, 100) # Coordinates for the bottom left point
    glVertex2f(200, 100) # Coordinates for the bottom right point
    glVertex2f(200, 200) # Coordinates for the top right point
    glVertex2f(100, 200) # Coordinates for the top left point
    glEnd() # Mark the end of drawing

# This alone isn't enough to draw our square

def draw_curve_polygon(points):
    glBegin(GL_LINE_STRIP) # Begin the sketch
    for point in points:
        glVertex2f(point[0], point[1])
    glEnd()

def draw_surface_polygon(color, points_matrix):
    glColor3f(*color)
    glBegin(GL_LINE_STRIP)
    for points in points_matrix:
        for point in points:
            glVertex2f(point[0], point[1])
    glEnd()


def draw_points(color, points):
    glColor3f(*color)
    glPointSize(8)
    glBegin(GL_POINTS)
    for point in points:
        glVertex2f(point[0], point[1])
    glEnd()

def show_curve(curve_points):
    glBegin(GL_LINE_STRIP) # Begin the sketch
    for point in curve_points:
        glVertex2f(point[0], point[1])
    glEnd()

# ---Section 2---

def iterate():
    glViewport(0, 0, 500,500)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0.0, 9.0, 0.0, 9.0, 0.0, 1.0)
    glMatrixMode (GL_MODELVIEW)
    glLoadIdentity()

def showScreen():
    global control_points
    global curve_points
    global extra_points

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) # Remove everything from screen (i.e. displays all white)
    glLoadIdentity() # Reset all graphic/shape's position
    iterate()
    glColor3f(1.0, 0.0, 3.0)
    #draw_surface_polygon((0,0,1), control_points)
    #glColor3f(1.0, 3.0, 3.0)
    #show_curve(curve_points)
    draw_points((1,0,0),surface_points)
    #draw_points((1,0,0), extra_points)
    draw_points((0,1,0), control_list)
    #square()
    glutSwapBuffers()

#---Section 3---

P0 = np.array((1,1,0))
P1 = np.array((2,1,0))
P2 = np.array((3,1,0))
P3 = np.array((1,3,0))
P4 = np.array((2,3,0))
P5 = np.array((3,3,0))
P6 = np.array((1,5,0))
P7 = np.array((2,5,0))
P8 = np.array((3,5,0))
#P9 = np.array((5,5,0))


global control_list
control_list = [P0, P1, P2, P3, P4, P5, P6, P7, P8]#, P9]
global control_points3
control_points = [[P0, P1, P2], [P3, P4, P5], [P6, P7, P8]]#, P9]]
'''
#curve = BezierCurve(control_points, resolution=0.01)
#curve = BezierCurve(control_points, resolution=0.01,  curve_type='q_spline', smoothness='c1')
#curve = BezierCurve(control_points, resolution=0.01,  curve_type='c_spline', smoothness='')

global curve_points
curve_points = curve.solve_curve()
global extra_points 
extra_points = curve.extra_points
print('extra points:')
print(extra_points)
'''

surface = BezierSurface(control_points, 2, 2)

global surface_points
surface_points = surface.solve_surface()
print (surface_points)


glutInit()
glutInitDisplayMode(GLUT_RGBA) # Set the display mode to be colored
glutInitWindowSize(500, 500)   # Set the w and h of your window
glutInitWindowPosition(0, 0)   # Set the position at which this windows should appear
wind = glutCreateWindow("Bezier") # Set a window title
glutDisplayFunc(showScreen)
glutIdleFunc(showScreen) # Keeps the window open
glutMainLoop()  # Keeps the above created window displaying/running in a loop