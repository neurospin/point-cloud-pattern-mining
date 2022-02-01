import numpy
import logging
log = logging.getLogger(__name__)


def parse_bucket_file(bucket_file):
    """
    Parse a bucket file

    return values:
        dxyz: numpy.array of shape (3,),  bucket scale coefficients (dx,dy,dz)
        data_points: numpy.array of shape (n,3), the data points
        dim: number, the value of the dim parameter
    """

    # number of header lines in a bucket file
    header_len = 9
    # >>> READ THE .bck (bucket) FILE
    lines = list()
    with open(bucket_file, 'r') as f:
        # read the lines and remove whitespaces and newlines
        lines = [line.strip() for line in f.readlines()]
        lines_n = len(lines)

        # >>> CHECK FOR EMPTY BUCKET FILES
        if lines_n < header_len:
            raise ValueError("The bucket file {} has less than 9 lines")
        elif (lines_n == header_len):  # The file header is 10 lines
            log.warn("The file {} appears to contain no data points ({} lines".format(
                lines_n))

    # get the coefficients from the file's header
    dxyz = numpy.zeros(3)
    for i, line in enumerate(lines[2:5]):
        dxyz[i] = float(line[4:])

    # get the dim parameter from bck file
    dim = lines[8][5:]

    # get the data points
    data_points = numpy.empty((lines_n-header_len, 3))
    for i, line in enumerate(lines[9:]):
        line = line[1:-1]  # remove brackets
        # read one data point
        data_points[i] = numpy.array(line.split(','), dtype=float)

    return dxyz, data_points, dim


def rescale_array_to_int16(x:numpy.ndarray):
    """Convert x to int8 with minimum precision loss without translation.
    
    The abs-max of the rescaled-x is equal to the max value
    representable as type int8.
    
    :param x: input array
    :type x: numpy.ndarray
    :return: rescaled array of type int16, scaling factor
    :rtype: -> (numpy.array, float)
    """
    dx = numpy.abs(x).max()/(numpy.iinfo(numpy.int8).max)
    x = x/dx
    return numpy.round(x).astype(int), dx


def save_bucket_file(fname: str,
                     points: numpy.ndarray,
                     bucket_type: str = 'VOID',
                     #dx: float = 1, dy: float = 1, dz: float = 1,
                     dt: float = 1,
                     dimt: float = 1, time: float = 0):
    """Save a bucket file with given points"""

    assert points.shape[1] == 3,\
        "points array should have shape Nx3, got{}".format(points.shape)

    # rescale the coordinates and calculate the scale factors
    x,y,z = points.T
    x, dx =  rescale_array_to_int16(x)
    y, dy =  rescale_array_to_int16(y)
    z, dz =  rescale_array_to_int16(z)

    points = numpy.array([x,y,z]).T

    header = [
        "ascii", "-type {}".format(bucket_type),
        "-dx {}".format(dx), "-dy {}".format(dy),
        "-dz {}".format(dz), "-dt {}".format(dt),
        "-dimt {}".format(dimt), "-time {}".format(time),
        "-dim {}".format(len(points))]

    fmt = "({}, {}, {})" 
    points_str = [fmt.format(*numpy.round(p).astype(int)) for p in points ]
    
    s = '\n'.join(header+points_str)
    with open(fname, 'w') as f:
        f.write(s)


def parse_transformation_file(transform_fname):
    """Get the transformation matrix from a .trm transformation file.

    :param transform_fname: the path to the transformation file
    :type transform_fname: str
    :return: (rotation matrix, translation vector)
    :rtype: tuple of nunmpy.ndarrays
    """

    transform_matrix = numpy.loadtxt(transform_fname, dtype=float)
    translation = transform_matrix[0]
    rotation = transform_matrix[1:]
    return rotation, translation


def save_transformation_file(fname: str, rot_M: numpy.ndarray, tra_v: numpy.ndarray):
    """Create a transformation file from given rotation matrix and 
    translation vector.

    :param fname: output file name
    :type fname: str
    :param rot_M: rotation matrix
    :type rot_M: numpy.ndarray
    :param tra_v: translation vector
    :type tra_v: numpy.ndarray
    """
    transf_M = numpy.empty((4,3), dtype=float)
    transf_M[0] = tra_v
    transf_M[1:] = rot_M
    numpy.savetxt(fname, transf_M, fmt='%.6f')