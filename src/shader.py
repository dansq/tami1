from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

class Shader:
    """Creates a shader using src files."""

    def __init__(self, vs_path, fs_path):
        #   breakpoint()

        with open(vs_path, 'r') as file:
            vertex_src = ''
            for line in file.readlines():
                vertex_src += line

        with open(fs_path, 'r') as file:
            fragment_src = ''
            for line in file.readlines():
                fragment_src += line

        self.shader = compileProgram(
            compileShader(vertex_src, GL_VERTEX_SHADER),
            compileShader(fragment_src, GL_FRAGMENT_SHADER),
        )

    def use(self):
        glUseProgram(self.shader)