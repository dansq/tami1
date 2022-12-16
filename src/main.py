import OpenGL
import OpenGL.GL as gl
import OpenGL.GLUT as glu
import OpenGL.GLU as glut

print('eitcahneorooeis')

def showScreen():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

glu.glutInit()
glu.glutInitDisplayMode(glu.GLUT_RGBA)
glu.glutInitWindowSize(500,500)
glu.glutInitWindowPosition(0,0)
window = glu.glutCreateWindow("ETCHANOIS")
glu.glutDisplayFunc(showScreen)
glu.glutIdleFunc(showScreen)
glu.glutMainLoop()