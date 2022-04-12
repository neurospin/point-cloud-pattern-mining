import numpy
import multiprocessing as mp


# class DistanceResult:
#     """Regrup the sandard result of ICP distance functions"""

#     def __init__(self, distance, rotation_M, translation_v, dist_fun_name):
#         self.distance = distance
#         self.rotation_M = rotation_M
#         self.translation_v = translation_v
#         self.dist_fun_name = dist_fun_name

#     def __repr__(self):
#         return "\nICP result: function = '{}' Distance: {:.2e}\n\
# Rotation:\n{}\nTranslation: {}".format(self.dist_fun_name,
#                                        self.distance,
#                                        numpy.array_str(self.rotation_M, precision=2),
#                                        numpy.array_str(self.translation_v, precision=2))


def _calc_closest_point(a1: numpy.ndarray, a2: numpy.ndarray):
    """Calculate the closest points of a2 relative to the points of a1.

    The shape of the array is interpreted as DxN where D is the
    number of the point's coordinates and N the number of points.

    :param a1: array against which the distance are calculated
    :type a1: DxN numpy.ndarray
    :param a2: second array
    :type a2: DxM numpy.ndarray
    :return: closest points coodrinates (DxN),
        closest points indices in a2 (1xN),
        average distance (scalar)
    :rtype: tuple
    """

    # Calculate the square distance of each point of a2 with all points of a1
    # NOTE: this is faster than scipy.spatial's distances
    # 1) calculate the square-difference of the coordinates
    dist = ((a2.T[..., None] - a1)**2)  # shape = (a2-n-points, 3, a1-n-points)
    # 2) sum the square differences of the corrdinate along the coordinates axis
    #    get the square-distance matrix
    dist = dist.sum(axis=1)  # shape = (a2-n-points, a1-n-points)

    # the distance of the closest points of a2 for each point in a1
    # ! OLD : self.minDist[:, 1] = dist.min(axis=0)
    closest_points_distances = dist.min(axis=0)  # shape (1, a1-n-points)

    # get the index of the closest point of a2 to each point of a1
    # ! OLD : self.minDist[:, 0] = indices
    closest_points_idx = dist.argmin(axis=0)  # shape = (1, a1-n-points)

    # TODO: the min is calculated twice in the previous expressions, it could be done only once

    # the coordinates of the points of a2 which are closest to a1
    # ! OLD : self.curClose = self.model[:, indices]
    closest_points = a2[:, closest_points_idx]

    # average distance
    # ! OLD : self.curDist = self.minDist[:,1].sum()/self.numPtr
    av_dist = closest_points_distances.mean()

    return closest_points, av_dist


def _calc_transform(a1: numpy.ndarray, a2: numpy.ndarray, closest_points: numpy.ndarray):
    """Calculate SVD-based rotation and translation that
    transform the points in a1 into those of a2"""

    # TODO this could be taken as fucntion argument, not calcuated every time
    meanModel = a2.mean(axis=1)[:, None]
    meanData = a1.mean(axis=1)[:, None]

    A = a1 - meanData
    B = closest_points - meanModel

    # SVD decomposition
    (U, S, V) = numpy.linalg.svd(numpy.dot(B, A.T))
    U[:, -1] *= numpy.linalg.det(numpy.dot(U, V))

    rotation_M = numpy.dot(U, V)
    translation_v = meanModel - numpy.dot(rotation_M, meanData)

    return rotation_M, translation_v


def icp_python(moving: numpy.ndarray, model: numpy.ndarray, max_iter: int = 10, epsilon: float = 0.1):
    """Calculate Iterative Closest Point (ICP) distance.

    Array a1 is iteratively rotated and translated to make it closer
    to array a2.

    Args:
        moving ndarray (N,3): point_cloud
        model ndarray (N,3): reference point_cloud
        max_iter (int, optional): man number of iterations, defaults to 10
        epsilon, float: min distance improvement on one iteration, defaults to 0.1

    Returns:
    (dist,rot,trans) tuple containing
    - the final average distance between closest points after transformation of a1
    - the rotation matrix and the translation vector of the ICP transformation of a1
    """

    a1 = moving.T
    a2 = model.T

    dim_n = a1.shape[0]  # single point dimensionality
    # cumulative rotation matrix and translation vector
    cum_rot = numpy.eye(dim_n, dim_n, dtype=float)
    cum_tra = numpy.zeros((dim_n, 1), dtype=float)

    old_dist = 0

    # check that max_iter is valid
    assert max_iter < numpy.iinfo(numpy.int).max
    max_iter = numpy.int(max_iter)

    for i in range(max_iter):
        # calculate distance
        pts, dist = _calc_closest_point(a1, a2)
        improvement = abs(old_dist - dist)

        if (improvement <= epsilon):
            break

        # calculate rotation and transformation
        rot, tra = _calc_transform(a1, a2, pts)

        # update the rotation and translation
        a1 = numpy.dot(rot, a1) + tra
        cum_rot = numpy.dot(rot, cum_rot)
        cum_tra = numpy.dot(rot, cum_tra) + tra

        # update the loop variables
        old_dist = dist
        i += 1

    return dist, cum_rot, cum_tra.flatten()


def simple(a1: numpy.ndarray, a2: numpy.ndarray):
    """ 
    calculate average distance of the closest points of a1 and a2
    """

    _, av_dist = _calc_closest_point(a1.T, a2.T)

    return av_dist
