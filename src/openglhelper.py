from OpenGL.GL import *

import numpy as np
import pyrr
import glfw


#region GLFW HELPER METHODS
def framebuffer_size_callback(window, w, h):
    glViewport(0,0,w,h)

def create_window(scr_w=640, scr_h=480):
    '''CREATING THE WINDOW'''

    window = glfw.create_window(scr_w, scr_h, "bezierzando", None, None)
    if not window:
        print('--- FAILURE TRYING TO CREATE WINDOW ---')
        glfw.terminate()
    
    glfw.make_context_current(window)
    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)
    return window

def process_input(window):
    if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
        glfw.set_window_should_close(window, True)
#endregion


#region MISC OPENGL HELPER METHODS
def setup_triangles_vertices(triangles:np.ndarray, shading='smooth', normals=False, use_colors=False, colors=False, has_texture=False, tex_coord=False):
    """formats points to a valid opengl VBO format"""
    if not normals:
        vbo_normals = compute_triangle_normals(triangles, shading=shading)
    if use_colors and not colors:
        # TODO, add something to use colors instead of textures
        pass
    if has_texture and not tex_coord:
        # we have a texture, but we need to compute the text coord for each vertex
        vbo_tex_coord = map_texture_to_triangles(triangles)
    
    vbo_data = []
    v_idx = 0
    for triangle in triangles:
        for vertex in triangle:
            aux_position = vertex.tolist() # 3 floats
            aux_tex_coord = vbo_tex_coord[v_idx].tolist() # 2 floats
            aux_normals = vbo_normals[v_idx].tolist() # 3 floats
            vbo_data.append(aux_position + aux_tex_coord + aux_normals)
            v_idx += 1
    
    vbo_data = np.array(vbo_data, dtype=np.float32)  

    return vbo_data

def compute_triangle_normals(triangles:np.ndarray, shading:str='flat', grouped_triangles=True) -> np.ndarray:
    """computes the normals for the triangles vertices
    
    shading can be either 'flat' or 'smooth'
    returns already in a list semi-compatible to VBO format"""

    # if it's flat shading, triangle_array should be a triangle
    # for phong (smooth) shading, triangle_array will be a list of triangles

    #getting each point A, B, C

    # each triangle has 3 vertices, each vertice has a position with 3 axis
    # size returns each element, so we would have n * 3 * 3 elements, that is the number of floats, not of vertices
    # so we just need to divide by three to get the number of vertices
    vertices_normals = np.zeros((triangles.size // 3, 3), dtype=np.float32)

    if shading == 'flat':
        v_idx = 0
        for triangle in triangles:
            triangle_normal = pyrr.vector3.normalize(
                compute_triangle_face_normal_vector(triangles)
                )
            for _ in triangle:
                vertices_normals[v_idx] = triangle_normal
                v_idx += 1

    if shading == 'smooth':  
        vertex_stride = 0
        if not grouped_triangles:
            while vertex_stride < len(triangles):
                # a triangle is made of three points
                try:
                    triangle = np.array(
                        [triangles[vertex_stride],
                        triangles[vertex_stride+1],
                        triangles[vertex_stride+2]], dtype=np.float32)

                    triangle_normal = compute_triangle_face_normal_vector(triangle)
                    vertices_normals[vertex_stride] += triangle_normal
                    vertices_normals[vertex_stride+1] += triangle_normal
                    vertices_normals[vertex_stride+2] += triangle_normal
                except IndexError:
                    # no more triangles
                    pass
                vertex_stride += 3
        else:
            for triangle in triangles:
                if type(triangle[0]) != np.ndarray:
                    #breakpoint()
                    pass
                
                triangle_normal = compute_triangle_face_normal_vector(triangle)
                vertices_normals[vertex_stride] += triangle_normal
                vertices_normals[vertex_stride+1] += triangle_normal
                vertices_normals[vertex_stride+2] += triangle_normal

                vertex_stride += 3

        
        for idx, normal in enumerate(vertices_normals):
            vertices_normals[idx] = pyrr.vector3.normalize(normal)
    return vertices_normals

def compute_triangle_face_normal_vector(triangle:np.ndarray) -> np.ndarray:
    """THIS IS NOT A UNIT VECTOR"""
    A = triangle[0]
    B = triangle[1]
    C = triangle[2]

    normal = pyrr.vector3.cross(B-A, C-A)
    return normal

def map_value_to_range(v, min_v, max_v, a, b):
    if max_v == min_v:
        min_v = min_v + 10**-50
    
    result = (b - a) * ((v - min_v)/(max_v - min_v)) + a
    return result

def normalize_point(point, max_values, min_values, a, b):
    n_x = map_value_to_range(point[0], min_values[0], max_values[0], a, b)
    n_y = map_value_to_range(point[1], min_values[1], max_values[1], a, b)
    n_z = map_value_to_range(point[2], min_values[2], max_values[2], a, b)

    return (n_x, n_y, n_z)

def get_max_and_min_values(triangles:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    "for texture uv mapping"
    max = ''
    min = ''

    for triangle in triangles:
        for point in triangle:
            if type(min) == np.ndarray:
                
                if point[0] < min[0]:
                    min[0] = point[0]
                if point[1] < min[1]:
                    min[1] = point[1]
                if point[2] < min[2]:
                    min[2] = point[2]
            else:
                min = np.array((point[0], point[1], point[2]), dtype=np.float32)
            
            if type(max) == np.ndarray:
                if point[0] > max[0]:
                    max[0] = point[0]
                if point[1] > max[1]:
                    max[1] = point[1]
                if point[2] > max[2]:
                    max[2] = point[2]
            else:
                max = np.array((point[0], point[1], point[2]), dtype=np.float32)

    return max, min

def map_texture_to_triangles(triangles:np.ndarray, grouped_triangle=True) -> np.array:
    """creates a list of texture coordinates"""

    #first we get the max and min values for x and y
    max_values, min_values = get_max_and_min_values(triangles)

    #now we get the corresponding coordinates
    normalized_coordinates = []

    if grouped_triangle:
        for triangle in triangles:
            for point in triangle:
                uv_coord = normalize_point(point, max_values, min_values, 0, 1)
                normalized_coordinates.append((uv_coord[0], uv_coord[1]))
    
    else:
        for point in triangles:
            uv_coord = normalize_point(point, max_values, min_values, 0, 1)
            normalized_coordinates.append((uv_coord[0], uv_coord[1]))
    
    
    normalized_coordinates = np.array(normalized_coordinates, dtype=np.float32)

    return normalized_coordinates

#endregion
