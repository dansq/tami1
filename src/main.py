from classes import App
from bezier import BezierSurface
import numpy as np

#region helper functions

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

#endregion

P0 = np.array((-1, -1, 0), dtype=np.float32)
P1 = np.array(( 0, -1, 3), dtype=np.float32)
P2 = np.array(( 1, -1, 1), dtype=np.float32)
P3 = np.array((-1, 0, -2), dtype=np.float32)
P4 = np.array(( 0, 0, 2), dtype=np.float32)
P5 = np.array(( 1, 0, 3), dtype=np.float32)
P6 = np.array((-1, 1, 4), dtype=np.float32)
P7 = np.array(( 0, 1, -2), dtype=np.float32)
P8 = np.array(( 1, 1, 1), dtype=np.float32)


global control_list
control_list = [P0, P1, P2, P3, P4, P5, P6, P7, P8]
global control_points
control_points = [[P0, P1, P2],
                  [P3, P4, P5],
                  [P6, P7, P8]]

print('control points')
for row in control_points:
    print(row)
print('---')

global surface
global surface_points

center = find_surface_center_point(control_points)
control_points = translate_to(control_points, -center)

print('Center')
print(center)
print('............')


print('centered control points')
for row in control_points:
    print(row)
print('---')

surface = BezierSurface(control_points, 2, 2, n_div=8)
print(f'surface attributes\nu_step: {surface.u_step}\nv_step: {surface.v_step} ')
#breakpoint()
surface_points = surface.solve_surface()



#surface.update_surface_points(surface_points)



#surface.update_surface_points(surface_points)
#print (surface_points)

global triangles
triangles = surface.find_triangles()

print(f"triangles:{surface.triangles.size//9}\n{surface.triangles}")
print(f'surface_points: {surface.surface_points.size//3}\n{surface.surface_points}')

if __name__ == "__main__":
    #breakpoint()
    app = App(surface=surface, tex_path='gfx/zelador.png')
    print("ACABOU")