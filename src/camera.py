import numpy as np

class Camera:

    def __init__(self, pos, up, dir) -> None:
        self.pos = pos
        self.up = up
        self.dir = dir
        #breakpoint()
        right = np.cross(up, dir)
        self.right = self.normalize(right)
        self.rot = ''
    
    def normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0: 
            return v
        return v / norm


    def get_look_at_matrix(self):
        # A * B

        A = np.array([
            (self.right[0], self.right[1], self.right[2], 0),
            (self.up[0],    self.up[1],    self.up[2], 0),
            (self.dir[0],   self.dir[1],   self.dir[2], 0),
            (0,             0,             0,             1)
        ])


        B = np.array([
            (0, 0, 0, -self.pos[0]),
            (0, 0, 0, -self.pos[1]),
            (0, 0, 0, -self.pos[2]),
            (0, 0, 0, 1)
        ])

        return A * B

    def translate(self, translation):
        new_pos = (self.pos[0] + translation[0],
                   self.pos[1] + translation[1],
                   self.pos[2] + translation[2]
                   )
        self.pos = new_pos

    def set_rotation(self, rot):
        self.rot = rot