import numpy as np

"""this is just for holding helper methods"""

def read_control_points_from_file(filepath, surface=True, points_per_row=3, separator=','):
    """similar to a csv, each line should have 3 numbers
    floats or ints should work, separator can be defined"""

    with open(filepath, 'r') as file:
        control_points = []

        points_in_row = 0
        if surface:
            temp_row = []
        for line in file.readlines():
            if surface:
                if points_in_row == points_per_row:
                    points_in_row = 0
                    temp_row = []
                    control_points.append(temp_row)
                
                split_line = line.split(sep=separator)

                temp_position = np.array((
                    float(split_line[0].strip()),
                    float(split_line[1].strip()),
                    float(split_line[2].strip()),), dtype=np.float32)
                temp_row.append(temp_position)
                points_in_row += 1
        
        control_points.append(temp_row)

    control_points = np.array(control_points, dtype=np.float32)

    return control_points

            

