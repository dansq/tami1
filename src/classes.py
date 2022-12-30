from OpenGL.GL import *
from shader import Shader
from PIL import Image

import numpy as np
import pyrr
import glfw
import ctypes

from bezier import BezierSurface
from openglhelper import setup_triangles_vertices, process_input, create_window

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
        #print(f'position: {self.position}\nforwards: {self.forwards}\nright: {self.right}\nup: {self.up}')
    
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
    
    def set_forwards(self, fwd:tuple):
        self.forwards = np.array(fwd, dtype=np.float32)
        #breakpoint()
        globalUp = np.array([0,0,1], dtype=np.float32)

        self.right = np.cross(self.forwards, globalUp)

        self.up = np.cross(self.right, self.forwards)
        #print(f'position: {self.position}\nforwards: {self.forwards}\nright: {self.right}\nup: {self.up}')
        

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
        self.vertex_count = len(self.vertices) # vertices are already encapsulated and float32

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        for idx, offset in enumerate(model_vbo.offsets):
            glEnableVertexAttribArray(idx)
            glVertexAttribPointer(idx,
                                  offset[0],
                                  GL_FLOAT,
                                  GL_FALSE,
                                  model_vbo.stride,
                                  ctypes.c_void_p(offset[1]))

    def destroy(self):
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1, (self.vbo,))

class Transforms:
    def __init__(self, position, eulers) -> None:
        self.position = np.array(position, dtype=np.float32)
        self.eulers = np.array(eulers, dtype=np.float32)

class ObjectContainer:
    """holds transforms, texture and a mesh"""

    def __init__(self, transform:Transforms, mesh:Mesh, texture:Material) -> None:
        self.mesh = mesh
        self.texture = texture
        self.transforms = transform

class World:
    """holds surface and lights"""

    def __init__(self, surface_vbo, texture):

        surface_transform = Transforms(
                position=[0,0.5,-3],
                eulers=[0,0,0]
            )

        surface_mesh = Mesh(surface_vbo)
        
        self.surface = ObjectContainer(
            transform=surface_transform,
            mesh=surface_mesh,
            texture=texture
            )

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
                strength = 10
            )
            for i in range(8)
        ]

        self.camera = Camera(
            position = [0,0,2]
        )

        #self.camera.set_forwards((0,1,0))

    def update(self, surface:ObjectContainer, rate, axis=2):
        #breakpoint()
        surface.transforms.eulers[axis] += 0.25 * rate
        if surface.transforms.eulers[axis] > 360:
            surface.transforms.eulers[axis] -= 360

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

    def __init__(self, surface:BezierSurface, tex_path='gfx/map_checker.png', verbose=False, camera_pos=(0,0,4), shading='smooth') -> None:
        """initialize the program"""
        # initialize glfw, create window, load shaders, etc
        self._start_context(shading=shading)

        # models, textures, other stuff
        surface_data = setup_triangles_vertices(surface.triangles, shading=shading, has_texture=True)
        if verbose:
            print(surface_data)

        surface_vbo = VBO(data=surface_data, stride=32, offsets=((3, 0),(2,12),(3,20)), elements_per_vertex=8)
        texture = Material(tex_path)
        world = World(surface_vbo, texture)



        projection_transform = pyrr.matrix44.create_perspective_projection(
            fovy=45, aspect=640/480,
            near=0.1, far=10, dtype=np.float32
        )

        #print(projection_transform)

        glUniformMatrix4fv(
            glGetUniformLocation(self.shader.shader, 'projection'),
            1, GL_FALSE, projection_transform
        )

        self.modelMatrixLocation = glGetUniformLocation(self.shader.shader, 'model')
        self.viewMatrixLocation = glGetUniformLocation(self.shader.shader, 'view')
        if shading == 'smooth':
            self.lightLocation = {
                "position": [
                    glGetUniformLocation(self.shader.shader, f"Lights[{i}].position")
                    for i in range(8)
                ],
                "color": [
                    glGetUniformLocation(self.shader.shader, f"Lights[{i}].color")
                    for i in range(8)
                ],
                "strength": [
                    glGetUniformLocation(self.shader.shader, f"Lights[{i}].strength")
                    for i in range(8)
                ]
            }
        #self.camera = Camera(camera_pos)
        self.cameraPosLoc = glGetUniformLocation(self.shader.shader, "cameraPosition")

        # main_loop
        self._render_loop(world, shading)

    def _start_context(self, shading='smooth'):
        glfw.init()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        self.window = create_window(1280, 960)

        glClearColor(0.1, 0.2, 0.2, 1.0)
        glEnable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        vs_path = r'shaders/simple/vertex.vs'
        fs_path = r'shaders/simple/fragment.fs'
        if shading == 'smooth':
            vs_path = r'shaders/phong/vertex.vs'
            fs_path = r'shaders/phong/fragment.fs' 

        self.shader = Shader(
            vs_path=vs_path,
            fs_path=fs_path
            )
        self.shader.use()

        glUniform1i(glGetUniformLocation(self.shader.shader, "imageTexture"), 0)

    def _render_loop(self, world:World, shading='smooth'):
        while not glfw.window_should_close(self.window):
            process_input(self.window)

            # update model rotation
            world.update(world.surface, 0.5)
            ''' self.square.eulers[2] += 0.2
            if self.square.eulers[2] > 360:
                self.square.eulers[2] -= 360'''

            # refresh
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            self.shader.use()   
            world.surface.texture.use()

            view_transform = pyrr.matrix44.create_look_at(
                eye=world.camera.position,
                target=world.camera.position + world.camera.forwards,
                up=world.camera.up, dtype=np.float32
            )

            #breakpoint()

            glUniformMatrix4fv(self.viewMatrixLocation, 1, GL_FALSE, view_transform)

            if shading == 'smooth':
                for i,light in enumerate(world.lights):
                    glUniform3fv(self.lightLocation["position"][i], 1, light.position)
                    glUniform3fv(self.lightLocation["color"][i], 1, light.color)
                    glUniform1f(self.lightLocation["strength"][i], light.strength)

            glUniform3fv(self.cameraPosLoc, 1, world.camera.position)
            
            # update matrices
            model_transform = pyrr.matrix44.create_identity(dtype=np.float32)
            #rotate
            model_transform = pyrr.matrix44.multiply(
                m1=model_transform,
                m2=pyrr.matrix44.create_from_eulers(
                    eulers=np.radians(world.surface.transforms.eulers),
                    dtype=np.float32
                )
            )
            #translate
            model_transform = pyrr.matrix44.multiply(
                m1=model_transform,
                m2=pyrr.matrix44.create_from_translation(
                    vec=world.surface.transforms.position,
                    dtype=np.float32
                )
            )

            glUniformMatrix4fv(self.modelMatrixLocation, 1, GL_FALSE, model_transform)

            glBindVertexArray(world.surface.mesh.vao)
            glDrawArrays(GL_TRIANGLES, 0, world.surface.mesh.vertex_count)

            glfw.swap_buffers(self.window)
            glfw.poll_events()

            

        world.surface.mesh.destroy()
        world.surface.texture.destroy()
        glDeleteProgram(self.shader.shader)  
        glfw.terminate()
        return None
#endregion
