from OpenGL.GL import *
from shader import Shader
from PIL import Image

import numpy as np
import pyrr
import glfw
import ctypes

from bezier import BezierSurface


#region GLFW HELPER METHODS

def framebuffer_size_callback(window, w, h):
    glViewport(0,0,w,h)

def create_window(scr_w=640, scr_h=480):
    '''CREATING THE WINDOW'''

    window = glfw.create_window(scr_w, scr_h, "bezierzando", None, None)
    if not window:
        print('Falha em criar janela')
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
        #breakpoint()
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
    #breakpoint()    

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
    #breakpoint()
    try:
        normal = pyrr.vector3.cross(B-A, C-A)
    except:
        breakpoint()
        pass
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
    #breakpoint()

    #now we get the corresponding coordinates
    normalized_coordinates = []

    if grouped_triangle:
        for triangle in triangles:
            for point in triangle:
                uv_coord = normalize_point(point, max_values, min_values, 0, 1)
                normalized_coordinates.append((uv_coord[0], uv_coord[1]))
                #breakpoint()
    
    else:
        for point in triangles:
            uv_coord = normalize_point(point, max_values, min_values, 0, 1)
            normalized_coordinates.append((uv_coord[0], uv_coord[1]))
    
    
    normalized_coordinates = np.array(normalized_coordinates, dtype=np.float32)

    return normalized_coordinates

#endregion

#region MISC TEST VARIABLES
square_points = np.array(
    [[-0.5, -0.5, -0.5],
     [ 0.5, -0.5, -0.5],
     [ 0.5,  0.5, -0.5],
     [ 0.5,  0.5, -0.5],
     [-0.5,  0.5, -0.5],
     [-0.5, -0.5, -0.5]]
    , dtype=np.float32)

cube_data = vertices = (
                -0.5, -0.5, -0.5, 0, 0, 1,0,0,
                 0.5, -0.5, -0.5, 1, 0, 1,0,0,
                 0.5,  0.5, -0.5, 1, 1, 1,0,0,

                 0.5,  0.5, -0.5, 1, 1, 1,0,0,
                -0.5,  0.5, -0.5, 0, 1, 1,0,0,
                -0.5, -0.5, -0.5, 0, 0, 1,0,0,

                -0.5, -0.5,  0.5, 0, 0, 1,0,0,
                 0.5, -0.5,  0.5, 1, 0, 1,0,0,
                 0.5,  0.5,  0.5, 1, 1, 1,0,0,

                 0.5,  0.5,  0.5, 1, 1, 1,0,0,
                -0.5,  0.5,  0.5, 0, 1, 1,0,0,
                -0.5, -0.5,  0.5, 0, 0, 1,0,0,

                -0.5,  0.5,  0.5, 1, 0, 1,0,0,
                -0.5,  0.5, -0.5, 1, 1, 1,0,0,
                -0.5, -0.5, -0.5, 0, 1, 1,0,0,

                -0.5, -0.5, -0.5, 0, 1, 1,0,0,
                -0.5, -0.5,  0.5, 0, 0, 1,0,0,
                -0.5,  0.5,  0.5, 1, 0, 1,0,0,

                 0.5,  0.5,  0.5, 1, 0, 1,0,0,
                 0.5,  0.5, -0.5, 1, 1, 1,0,0,
                 0.5, -0.5, -0.5, 0, 1, 1,0,0,

                 0.5, -0.5, -0.5, 0, 1, 1,0,0,
                 0.5, -0.5,  0.5, 0, 0, 1,0,0,
                 0.5,  0.5,  0.5, 1, 0, 1,0,0,

                -0.5, -0.5, -0.5, 0, 1, 1,0,0,
                 0.5, -0.5, -0.5, 1, 1, 1,0,0,
                 0.5, -0.5,  0.5, 1, 0, 1,0,0,

                 0.5, -0.5,  0.5, 1, 0, 1,0,0,
                -0.5, -0.5,  0.5, 0, 0, 1,0,0,
                -0.5, -0.5, -0.5, 0, 1, 1,0,0,

                -0.5,  0.5, -0.5, 0, 1, 1,0,0,
                 0.5,  0.5, -0.5, 1, 1, 1,0,0,
                 0.5,  0.5,  0.5, 1, 0, 1,0,0,

                 0.5,  0.5,  0.5, 1, 0, 1,0,0,
                -0.5,  0.5,  0.5, 0, 0, 1,0,0,
                -0.5,  0.5, -0.5, 0, 1, 1,0,0,
            )
#endregion


#region CLASSES
class Light:

    def __init__(self, position, color, strength) -> None:
        self.position = np.array(position, dtype=np.float32)
        self.color = np.array(color, dtype=np.float32)
        self.strength = strength

class Camera:

    def __init__(self, position):

        self.position = np.array(position, dtype = np.float32)
        self.theta = 0
        self.phi = 0
        self.update_vectors()
    
    def update_vectors(self):

        self.forwards = np.array(
            [
                np.cos(np.deg2rad(self.theta)) * np.cos(np.deg2rad(self.phi)),
                np.sin(np.deg2rad(self.theta)) * np.cos(np.deg2rad(self.phi)),
                np.sin(np.deg2rad(self.phi))
            ],
            dtype = np.float32
        )

        globalUp = np.array([0,0,1], dtype=np.float32)

        self.right = np.cross(self.forwards, globalUp)

        self.up = np.cross(self.right, self.forwards)



class Material:

    def __init__(self, filepath) -> None:
        
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        with Image.open(filepath) as image:

            image_width, image_height = image.size
            image = image.transpose(Image.FLIP_TOP_BOTTOM)

            image = image.convert("RGBA")
            image_data = bytes(image.tobytes())
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0 , GL_RGBA, GL_UNSIGNED_BYTE, image_data)

        glGenerateMipmap(GL_TEXTURE_2D)

    def use(self):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)

    def destroy(self):
        glDeleteTextures(1,(self.texture,))

class VBO:
    """class to help with VBO management"""

    def __init__(self, data, stride, offsets, elements_per_vertex) -> None:
        self.data = data
        self.stride = stride
        #offsets[i] = (n_positions, byte_offset)
        self.offsets = offsets
        self.elements_per_vertex = elements_per_vertex
        pass

    def add_id(self, id):
        self.id = id 

class Mesh:

    def __init__(self, model_vbo:VBO) -> None:
        """creates a mesh from vertices stored in a numpy-like array, using the VBO class"""

        # |x:y:z|u:v|nx:ny:nz|
        self.vertices = model_vbo.data
        self.vertex_count = len(self.vertices)# // model_vbo.elements_per_vertex
        #breakpoint()

        #make sure data is already a np.float32 array
        #self.vertices = np.array(self.vertices, dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        for idx, offset in enumerate(model_vbo.offsets):
            glEnableVertexAttribArray(idx) # where are the positions on the buffer
            glVertexAttribPointer(idx,
                                  offset[0],
                                  GL_FLOAT,
                                  GL_FALSE,
                                  model_vbo.stride,
                                  ctypes.c_void_p(offset[1]))
        
        #glEnableVertexAttribArray(1) # where are the colors on the buffer
        #glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
        #glEnableVertexAttribArray(1) # where are the texture coordinates on the buffer
        #glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(12))

    def destroy(self):
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1, (self.vbo,))

class Transforms:
    def __init__(self, position, eulers) -> None:
        self.position = np.array(position, dtype=np.float32)
        self.eulers = np.array(eulers, dtype=np.float32)

class ObjectContainer:
    """holds transforms and a mesh"""

    def __init__(self, transform, mesh) -> None:
        self.mesh = mesh
        self.transforms = transform

class World:
    """holds surface and lights"""

    def __init__(self, surface):

        self.surface = [
            Transforms(
                position = [0,0,0],
                eulers = [0,0,0]
            ),
        ]

        self.lights = [
            Light(
                position = [
                    np.random.uniform(low=3.0, high=9.0), 
                    np.random.uniform(low=-2.0, high=2.0), 
                    np.random.uniform(low=2.0, high=4.0)
                ],
                color = [
                    np.random.uniform(low=0.5, high=1.0), 
                    np.random.uniform(low=0.5, high=1.0), 
                    np.random.uniform(low=0.5, high=1.0)
                ],
                strength = 3
            )
        ]

        self.camera = Camera(
            position = [0,0,2]
        )

    def update(self, rate):

        for cube in self.cubes:
            cube.eulers[1] += 0.25 * rate
            if cube.eulers[1] > 360:
                cube.eulers[1] -= 360

    def move_camera(self, dPos):

        dPos = np.array(dPos, dtype = np.float32)
        self.camera.position += dPos
    
    def spin_camera(self, dTheta, dPhi):

        self.camera.theta += dTheta
        if self.camera.theta > 360:
            self.camera.theta -= 360
        elif self.camera.theta < 0:
            self.camera.theta += 360
        
        self.camera.phi = min(
            89, max(-89, self.camera.phi + dPhi)
        )
        self.camera.update_vectors()

class App:

    def __init__(self, surface:BezierSurface, tex_path='gfx/map_checker.png') -> None:
        """initialize the program"""
        # initialize glfw, create window, load shaders, etc
        self._start_context()

        # models, textures, other stuff
        self.square = Transforms(
            position=[0,0.5,-3],
            eulers=[0,0,0]
        )

        surface_data = setup_triangles_vertices(surface.triangles, shading='smooth', has_texture=True)
        print(surface_data)
        #breakpoint()
        #square_data = setup_triangles_vertices(square_points, shading='smooth', has_texture=True)

        square_vbo = VBO(data=surface_data, stride=32, offsets=((3, 0),(2,12),(3,20)), elements_per_vertex=8)

        self.square_mesh = Mesh(square_vbo)
        self.texture = Material(tex_path)

        projection_transform = pyrr.matrix44.create_perspective_projection(
            fovy=65, aspect=640/480,
            near=0.1, far=10, dtype=np.float32
        )

        glUniformMatrix4fv(
            glGetUniformLocation(self.shader.shader, 'projection'),
            1, GL_FALSE, projection_transform
        )

        self.modelMatrixLocation = glGetUniformLocation(self.shader.shader, 'model')

        # main_loop
        self._render_loop()

    def _start_context(self):
        glfw.init()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        self.window = create_window()

        glClearColor(0.1, 0.2, 0.2, 1.0)
        glEnable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.shader = Shader(r'shaders/simple/vertex.vs',
                        r'shaders/simple/fragment.fs'
                        )
        self.shader.use()

        glUniform1i(glGetUniformLocation(self.shader.shader, "imageTexture"), 0)

    def _render_loop(self):
        while not glfw.window_should_close(self.window):
            process_input(self.window)

            # update model rotation
            self.square.eulers[2] += 0.2
            if self.square.eulers[2] > 360:
                self.square.eulers[2] -= 360

            # refresh
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            self.shader.use()   
            self.texture.use()
            
            # update matrices
            model_transform = pyrr.matrix44.create_identity(dtype=np.float32)
            #rotate
            model_transform = pyrr.matrix44.multiply(
                m1=model_transform,
                m2=pyrr.matrix44.create_from_eulers(
                    eulers=np.radians(self.square.eulers),
                    dtype=np.float32
                )
            )
            #translate
            model_transform = pyrr.matrix44.multiply(
                m1=model_transform,
                m2=pyrr.matrix44.create_from_translation(
                    vec=self.square.position,
                    dtype=np.float32
                )
            )

            glUniformMatrix4fv(self.modelMatrixLocation, 1, GL_FALSE, model_transform)

            glBindVertexArray(self.square_mesh.vao)
            glDrawArrays(GL_TRIANGLES, 0, self.square_mesh.vertex_count)

            glfw.swap_buffers(self.window)
            glfw.poll_events()

            

        self.square_mesh.destroy()
        self.texture.destroy()
        glDeleteProgram(self.shader.shader)  
        glfw.terminate()
        return None

#endregion