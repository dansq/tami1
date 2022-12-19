from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

from bezier import BezierCurve, BezierSurface
from camera import Camera
import numpy as np
import keyboard as kb
import math

w, h = 500,500

# ---Section 1---

def draw_surface(color, triangles):
    glColor3f(*color)
    glBegin(GL_TRIANGLES)
    for triangle in triangles:
        for point in triangle:
            glVertex2f(point[0], point[1])
    glEnd()


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

def rotate_point_around_axis(point, axis, angle):
    #breakpoint()
    if axis == 'x':
        x_rot_matrix = np.array((
            (1, 0,               0),
            (0, math.cos(angle), -math.sin(angle)),
            (0, math.sin(angle), math.cos(angle))
            ))
        new_p = np.dot(x_rot_matrix, point)
    #breakpoint
    if axis == 'y':
        y_rot_matrix = np.array((
            (math.cos(angle),  0, math.sin(angle)),
            (0,                1, 0),
            (-math.sin(angle), 0, math.cos(angle))
            ))
        new_p = np.dot(y_rot_matrix, point)

    if axis == 'z':
        z_rot_matrix = np.array((
            (math.cos(angle), -math.sin(angle), 0),
            (math.sin(angle), math.cos(angle),  0),
            (0,               0,                1)
            ))
        new_p = np.dot(z_rot_matrix, point)

    print(new_p)
    return new_p

def rotate_surface(points_matrix, axis, angle):
    rotated_surface = []
    for row in points_matrix:
        rotated_row = []
        for point in row:
            rotated_row.append(rotate_point_around_axis(point, axis, angle))
        rotated_surface.append(rotated_row)

    return rotated_surface
            


def normalize_to_interval(v, min_v, max_v, a, b):
    #breakpoint()f
    if max_v == min_v:
        min_v = min_v + 10**-100
    return (b - a) * ((v - min_v)/(max_v - min_v)) + a

def normalize_point(point, max_values, min_values, a, b):
    #breakpoint()
    n_x = normalize_to_interval(point[0], min_values[0], max_values[0], a, b)
    n_y = normalize_to_interval(point[1], min_values[1], max_values[1], a, b)
    n_z = normalize_to_interval(point[2], min_values[2], max_values[2], a, b)

    return (n_x, n_y, n_z)

def get_max_and_min_values(points_matrix):
    max = ''
    min = ''

    for row in points_matrix:
        for point in row:
            #breakpoint()
            if type(min) == np.ndarray:
                
                if point[0] < min[0]:
                    min[0] = point[0]
                if point[1] < min[1]:
                    min[1] = point[1]
                if point[2] < min[2]:
                    min[2] = point[2]
            else:
                min = np.array((point[0], point[1], point[2]))
            
            if type(max) == np.ndarray:
                if point[0] > max[0]:
                    max[0] = point[0]
                if point[1] > max[1]:
                    max[1] = point[1]
                if point[2] > max[2]:
                    max[2] = point[2]
            else:
                max = np.array((point[0], point[1], point[2]))
    
    return max, min


def normalize_surface_points(points_matrix):
    #breakpoint()
    normalized_points_matrix = []
    max_values, min_values = get_max_and_min_values(points_matrix)
    
    for row in points_matrix:
        normalized_row = []
        for point in row:
            normalized_row.append(normalize_point(point, max_values, min_values, -1, 1))
        normalized_points_matrix.append(normalized_row)
    
    return normalized_points_matrix

def draw_surface_polygon(color, points_matrix):
    glColor3f(*color)
    glBegin(GL_LINE_STRIP)
    for points in points_matrix:
        for point in points:
            glVertex2f(point[0], point[1])
    glEnd()

def draw_point_grid(color, point_grid, size=4):
    glColor3f(*color)
    glPointSize(size)
    glBegin(GL_POINTS)
    for row in point_grid:
        for point in row:
            try:
                glVertex2f(point[0], point[1])
            except:
                breakpoint()
    glEnd()

def draw_points(color, points, size=4):
    glColor3f(*color)
    glPointSize(size)
    glBegin(GL_POINTS)
    for point in points:
        glVertex2f(point[0], point[1])
    glEnd()

def show_curve(curve_points):
    glBegin(GL_LINE_STRIP) # Begin the sketch
    for point in curve_points:
        glVertex2f(point[0], point[1])
    glEnd()

def find_surface_center_point(points_matrix):
    center_point  = 0.0
    k = 0
    for row in points_matrix:
        for point in row:
            center_point += point
            k += 1
    
    return center_point / k

def translate_to(points_matrix, translation):
    #breakpoint()
    new_positions = []
    for row in points_matrix:
        new_row = []
        for point in row:
            new_row.append(point + translation)
        new_positions.append(new_row)
    
    return new_positions
            

def rotate():
    global surface_points
    angle = 5.0
    speed = 0.001
    if kb.is_pressed('q'):
        surface_points = rotate_surface(surface_points, 'x', 5.0 * speed) 
    if kb.is_pressed('a'):
        surface_points = rotate_surface(surface_points, 'x', -5.0 * speed)
    if kb.is_pressed('w'):
        surface_points = rotate_surface(surface_points, 'y', 5.0 * speed)
    if kb.is_pressed('s'):
        surface_points = rotate_surface(surface_points, 'y', -5.0 * speed)
    if kb.is_pressed('e'):
        surface_points = rotate_surface(surface_points, 'z', 5.0 * speed)
    if kb.is_pressed('d'):
        surface_points = rotate_surface(surface_points, 'z', -5.0 * speed)

def update_camera(camera):
    translation = ''#move()
    
    camera.translate(translation)
    

    view = camera.get_look_at_matrix()
    glUniformMatrix4fv(np.array(camera.pos, dtype=float), 1, GL_FALSE, view)

# ---Section 2---

def iterate():
    glViewport(0, 0, 500,500)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-5., 5., -5., 5., -5., 5.)
    glMatrixMode (GL_MODELVIEW)
    glLoadIdentity()

def showScreen():
    global control_points
    global surface_points
    global curve_points
    global extra_points
    global triangles
    global camera
    global surface

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) # Remove everything from screen (i.e. displays all white)
    glLoadIdentity() # Reset all graphic/shape's position
    #update_camera(camera)
    iterate()
    rotate()
    surface.update_surface_points(surface_points)
    triangles = surface.find_triangles()
    glColor3f(1.0, 0.0, 3.0)
    #draw_surface_polygon((0,0,1), control_points)
    #glColor3f(1.0, 3.0, 3.0)
    #show_curve(curve_points)
    #draw_points((0,1,0), control_list, size=6)
    draw_point_grid((1,0,0),surface_points, size=3)
    draw_surface((0.5, 0.3, 0.2), triangles)
    #draw_points((1,0,0), extra_points)
    
    #square()
    glutSwapBuffers()

#---Section 3---

up = (0,1,0)
direction = (0,0,0)
pos = (0,0,-3)
global  camera
camera = Camera(pos, up, direction)




P0 = np.array((1,1,3))
P1 = np.array((2,1,1))
P2 = np.array((3,1,2))
P3 = np.array((1,3,5))
P4 = np.array((2,3,4))
P5 = np.array((3,3,7))
P6 = np.array((1,5,1))
P7 = np.array((2,5,2))
P8 = np.array((3,5,2))
#P9 = np.array((5,5,0))


global control_list
control_list = [P0, P1, P2, P3, P4, P5, P6, P7, P8]#, P9]
global control_points3
control_points = [[P0, P1, P2],
                  [P3, P4, P5],
                  [P6, P7, P8]]#, P9]]

print('control points')
for row in control_points:
    print(row)
print('---')

normalized_surface_cp = normalize_surface_points(control_points)

print('normalized control points')
for row in normalized_surface_cp:
    print(row)
print('---')
breakpoint
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

global surface
surface = BezierSurface(control_points, 2, 2)

global surface_points
surface_points = surface.solve_surface()
center = find_surface_center_point(surface_points)
surface_points = translate_to(surface_points, -center)

surface.update_surface_points(surface_points)

print('Center')
print(center)
print('............')

surface.update_surface_points(surface_points)
#print (surface_points)

global triangles
triangles = surface.find_triangles()

def rotate_z(_):
    global surface_points
    global surface
    surface_points = rotate_surface(surface_points, 'z', .05)
    surface.update_surface_points(surface_points)
    print(surface_points)
    

#kb.on_press_key('e', rotate_z)

glutInit()
glutInitDisplayMode(GLUT_RGBA) # Set the display mode to be colored
glutInitWindowSize(500, 500)   # Set the w and h of your window
glutInitWindowPosition(0, 0)   # Set the position at which this windows should appear
wind = glutCreateWindow("Bezier") # Set a window title
glutDisplayFunc(showScreen)
glutIdleFunc(showScreen) # Keeps the window open
glutMainLoop()  # Keeps the above created window displaying/running in a loop