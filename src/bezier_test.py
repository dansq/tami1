from bezier import BezierCurve
import numpy as np

P0 = np.array((1,0,0))
P1 = np.array((2,3,0))
P2 = np.array((8,4,0))
P3 = np.array((5,7,0))

control_points = [P0, P1, P2, P3]

curve = BezierCurve(control_points, type='basic')

curve_points = curve.solve_curve(0.01)

print(len(curve_points))
print(curve_points)