from classes import App
from bezier import BezierSurface
import numpy as np

from reader import read_control_points_from_file
import argparse

#region ARGUMENT PARSER SETUP
parser = argparse.ArgumentParser(description="Bézier surface drawing application")
parser.add_argument('-m', metavar='int', type=int, default=2, help='m degree, default is 2')
parser.add_argument('-n', metavar='int', type=int, default=2, help='n degree, default is 2')
parser.add_argument('-s', '--subdivisions', metavar='int', type=int, default=8, help='number of subdivisions, default is 8')
parser.add_argument('-cp', '--control_points', metavar='filepath', default='./control_points', type=str, help='filepath to the control points file')
parser.add_argument('-tp', '--texture_path', metavar='filepath', default='gfx/map_checker.png', type=str, help='filepath to the texture file')
parser.add_argument('-ka', metavar='float', type=float, default = 0.5, help='ambient coefficient for lighting')
parser.add_argument('-ks', metavar='float', type=float, default = 0.5, help='sepcular coefficient for lighting')
parser.add_argument('-kd', metavar='float', type=float, default = 0.5, help='diffuse coefficient for lighting')
parser.add_argument('-a', '--alfa', metavar='int', type=int, default = 16, help='alfa value for specular highlights')
parser.add_argument('-c', '--center', action='store_true', help='center the surface at the origin')
parser.add_argument('-v', '--verbose', action='store_true', help='activate verbose mode')

args = parser.parse_args()
#endregion

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

#region EXECUTION
verbose = args.verbose
center_bool = args.center
n_div = args.subdivisions
control_points = read_control_points_from_file(args.control_points)
tex_path = args.texture_path
l_const = (args.ka, args.kd, args.ks, args.alfa)

if center_bool:
    center = find_surface_center_point(control_points)
    oringal_cp = control_points
    control_points = translate_to(control_points, -center)
        
surface = BezierSurface(control_points, 2, 2, n_div=n_div)

if verbose:
    if center_bool:
        print('control points')
        for row in oringal_cp:
            print(row)
        print('---')

        print('Center')
        print(center)
        print('............')


        print('centered control points')
        for row in control_points:
            print(row)
        print('---')


    print(f'surface attributes\nu_step: {surface.u_step}\nv_step: {surface.v_step} ')

surface_points = surface.solve_surface()

triangles = surface.find_triangles()

if verbose:
    print(f"triangles:{surface.triangles.size//9}\n{surface.triangles}")
    print(f'surface_points: {surface.surface_points.size//3}\n{surface.surface_points}')
#endregion


if __name__ == "__main__":
    #breakpoint()
    app = App(surface=surface, tex_path=tex_path, verbose=verbose, l_const=l_const)
    print("ACABOU")
