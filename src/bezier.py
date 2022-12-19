import math


class BezierCurve:

    def __init__(self, control_points, resolution=0.1, curve_type='', smoothness=''):
        self.control_points = control_points
        self.resolution = resolution
        self.curve_type = curve_type
        self.smoothness = smoothness
        self.extra_points = ''
        self.surface_points = ''

    def find_c1_points(self):
        new_cp = []
        if self.curve_type == 'q_spline':
            idx = 1
            new_cp.append(self.control_points[0])
            while idx < len(self.control_points):
                if len(self.control_points) - idx == 2:
                    new_cp.append(self.control_points[idx])
                else:
                    try:
                        temp_cp = (self.control_points[idx] + self.control_points[idx+1])/2
                        new_cp.append(self.control_points[idx])
                        new_cp.append(temp_cp)
                    except IndexError:
                        print('eita')
                idx += 1
            new_cp.append(self.control_points[-1])

        if self.curve_type == 'c_spline':
            idx = 2
            new_cp.append(self.control_points[0])
            while idx < len(self.control_points):
                if len(self.control_points) - idx == 2:    
                    pass
                else:
                    try:
                        temp_cp = (self.control_points[idx] + self.control_points[idx+1])/2
                        new_cp.append(self.control_points[idx-1])
                        new_cp.append(self.control_points[idx])
                        new_cp.append(temp_cp)
                    except IndexError:
                        print('eita')
                idx += 2
            new_cp.append(self.control_points[-3])
            new_cp.append(self.control_points[-2])
            new_cp.append(self.control_points[-1])
        return new_cp

    def solve_curve_explicit(self, points=None):
        curve = []
        if not points:
            points = self.control_points
        n = len(points) - 1
        t = 0.0
        while t <= 1.0:
            temp_sum = 0.0
            for i in range(len(points)):
                temp_value =  math.comb(n, i) * math.pow((1-t), (n-i)) * math.pow(t, i) * points[i]
                temp_sum += temp_value
            B_t = temp_sum
            curve.append(B_t)
            t += self.resolution
        
        #if self.control_points[-1] in curve:
        #   pass
        #else:
        curve.append(points[-1])
        
        return curve

    def solve_cubic_spline(self):
        idx = 0
        full_curve = []
        if self.smoothness == 'c1' or self.smoothness == 'c2':
            points = self.extra_points
        else: 
            points = self.control_points
        
        while idx < len(points):
            print(idx, idx + 1, idx + 2, idx + 3)
            try:
                cubic_cp = [points[idx],
                                points[idx + 1],
                                points[idx + 2],
                                points[idx + 3]]
                print(cubic_cp)
                spline = self.solve_curve_explicit(points=cubic_cp)
                for point in spline:
                    full_curve.append(point)
            except IndexError:
                print('errou')
            idx += 3
        return full_curve

    def solve_quadratic_spline(self):
        idx = 0
        full_curve = []
        if self.smoothness == 'c1' or self.smoothness == 'c2':
            points = self.extra_points
        else: 
            points = self.control_points

        while idx < len(points):
            print(idx, idx + 1, idx + 2)
            try:
                quadratic_cp = [points[idx],
                                points[idx + 1],
                                points[idx + 2]]
                print(quadratic_cp)
                spline = self.solve_curve_explicit(points=quadratic_cp)
                for point in spline:
                    full_curve.append(point)
            except IndexError:
                print('errou')
            idx += 2
        return full_curve

    def solve_curve(self):
        # B(t) = SUM(i=0, n) {bi_comb(n, i) * (1-t)^(n-i)*t^(i)*P_()}
        curve = []

        if self.curve_type == 'c_spline':
            if self.smoothness == 'c1':
                self.extra_points = self.find_c1_points()
            elif self.smoothness == 'c2':
                pass

            curve = self.solve_cubic_spline()
        elif self.curve_type == 'q_spline':
            if self.smoothness == 'c1':
                self.extra_points = self.find_c1_points()
            elif self.smoothness == 'c2':
                pass

            curve = self.solve_quadratic_spline()
        else:
            curve = self.solve_curve_explicit()

        return curve
        

class BezierSurface:

    def __init__(self, control_points, m, n, resolution=0.1, n_div='') -> None:
        self.control_points = control_points
        self.m = m
        self.n = n
        self.resolution = resolution
        self.n_div = n_div

    def find_points_uv(self, n, m, u, v):
        # p(u,v) = SUM_{i=0}^{n} SUM_{j=0}^{n} B_{i}^{n}(u) B_{j}^{m}(v) k_{ij}
        # where B_{n}^{i}(u) = comb(n, i) u^{i} (1-u)^{n-i}
        result = 0.0
        for i in range(n+1):
            for j in range(m+1):
                #breakpoint()
                result += self.solve_B_u(u, n, i) * self.solve_B_u(v, m, j) * self.control_points[i][j]
        
        return result

    def solve_B_u(self, u, n, i):
        return math.comb(n, i) * math.pow(u, i) * math.pow((1 - u), (n-i))

    def find_triangles(self):
        '''
        points surface_points AxB
        '''
        x = 0

        all_triangles = []

        for row in self.surface_points:
            y = 0
            for column in row:
                try:
                    triangle_bot = [
                        self.surface_points[x][y],
                        self.surface_points[x+1][y],
                        self.surface_points[x+1][y+1],
                        ]
                    
                    triangle_top = [
                        self.surface_points[x][y],
                        self.surface_points[x][y+1],
                        self.surface_points[x+1][y+1],
                        ]
                except IndexError:
                    pass
                y += 1

                all_triangles.append(triangle_bot)
                all_triangles.append(triangle_top)

            x += 1

        return all_triangles


    def update_surface_points(self, points):
        self.surface_points = points

    def solve_surface(self):
        u = 0.0
        #breakpoint()
        surface_points = []
        while u <= 1.0:
            v = 0.0
            row_surface_points = []
            while v <= 1.0:
                row_surface_points.append(self.find_points_uv(self.n, self.m, u, v))
                #breakpoint
                v += 0.1 #self.resolution # v += 0.1
            surface_points.append(row_surface_points)
            #breakpoint()
            u += 0.1#self.resolution
        self.surface_points = surface_points
        return surface_points