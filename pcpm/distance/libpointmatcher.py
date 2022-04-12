try:
    from pypointmatcher import pointmatcher as pm, pointmatchersupport as pms
    HAS_LIBPOINTMATCHER = True
except ImportError as e:
    HAS_LIBPOINTMATCHER = False

import functools
from selectors import EpollSelector
import numpy as np


def _with_libpointmatcher(fun):
    @functools.wraps(fun)
    def wrapper(*args, **kwargs):
        if not HAS_LIBPOINTMATCHER:
            raise RuntimeError(
                "libpointmatcher is not available.")
        return fun(*args, **kwargs)
    return wrapper


if HAS_LIBPOINTMATCHER:
    PM = pm.PointMatcher
    DP = PM.DataPoints
    Parameters = pms.Parametrizable.Parameters


@_with_libpointmatcher
def np2dp(a):
    labels = DP.Labels()
    return DP(np.vstack([a.T, np.ones(a.shape[0])]), labels)


@_with_libpointmatcher
def dp2np(dp):
    return dp.features[:dp.features.shape[0]-1].T


@_with_libpointmatcher
def get_ICP_object(epsilon=0.1, max_iter=10):
    """Generate a custom ICP object using the libPointMatcher library.
    the ICP object represents the ICP pipeline.

    Args:
        epsilon (float, optional): min distance improvement on one iteration.
        max_iter (int, optional): stop at this iteration.

    Returns:
        pypointmatcher ICP object
    """
    icp = PM.ICP()

    # Prepare error minimization
    params = Parameters()
    name = "PointToPointErrorMinimizer"
    pointToPoint = PM.get().ErrorMinimizerRegistrar.create(name)
    params.clear()
    icp.errorMinimizer = pointToPoint

    # Prepare matching function
    name = "KDTreeMatcher"
    params["knn"] = "1"
    params["epsilon"] = str(epsilon)
    kdtree = PM.get().MatcherRegistrar.create(name, params)
    params.clear()
    icp.matcher = kdtree

    # Prepare inspector
    # Comment out to write vtk files per iteration
    name = "NullInspector"
    nullInspect = PM.get().InspectorRegistrar.create(name)
    icp.inspector = nullInspect

    # Prepare transformation
    name = "RigidTransformation"
    rigid_trans = PM.get().TransformationRegistrar.create(name)
    icp.transformations.append(rigid_trans)

    # Prepare transformation checker filters
    name = "CounterTransformationChecker"
    params["maxIterationCount"] = str(max_iter)
    max_iter = PM.get().TransformationCheckerRegistrar.create(name, params)
    params.clear()
    icp.transformationCheckers.append(max_iter)

    name = "DifferentialTransformationChecker"
    params["minDiffRotErr"] = "0.001"
    params["minDiffTransErr"] = "0.01"
    params["smoothLength"] = "4"
    diff = PM.get().TransformationCheckerRegistrar.create(name, params)
    params.clear()
    icp.transformationCheckers.append(diff)

    return icp


_DEFAULT_ICP = get_ICP_object() if HAS_LIBPOINTMATCHER else None


@_with_libpointmatcher
def set_default_icp_parameters(epsilon: float, max_iter: int):
    """set epsilon and max_iter of the default icp object.

    Args:
        epsilon (float, optional): min distance improvement on one iteration.
        max_iter (int, optional): stop at this iteration.
    """
    params = Parameters()

    # set epsilon
    name = "KDTreeMatcher"
    params["knn"] = "1"
    params["epsilon"] = str(epsilon)
    kdtree = PM.get().MatcherRegistrar.create(name, params)
    params.clear()
    _DEFAULT_ICP.matcher = kdtree

    # set max_iter
    name = "CounterTransformationChecker"
    params["maxIterationCount"] = str(max_iter)
    max_iter = PM.get().TransformationCheckerRegistrar.create(name, params)
    params.clear()
    _DEFAULT_ICP.transformationCheckers.append(max_iter)


@_with_libpointmatcher
def set_default_icp(icp_object):
    """Set the default ico object"""
    # TODO: assert type
    _DEFAULT_ICP = icp_object


@_with_libpointmatcher
def get_default_icp():
    return _DEFAULT_ICP


@_with_libpointmatcher
def icp_libpointmatcher(moving: np.ndarray, model: np.ndarray, icp_object=_DEFAULT_ICP, **kwargs):
    """Calculate Iterative Closest Point (ICP) distance.

    Args:
        moving (np.ndarray): point-cloud that will be moved
        model (np.ndarray): reference point-cloud
        icp_object (pypointmatcher icp object, optional): the ICP objects defining the pipeline.

    Returns:
        tuple : distance, rotation_matrix, translation_vector
    """

    d = calc_distance(moving, model, icp_object=icp_object)
    T = d['transformation']
    return d['dist'], T[0:3, 0:3], T[0:3, 3]


@_with_libpointmatcher
def calc_distance(pc, ref_pc, pre_transform=None, icp_object=_DEFAULT_ICP):
    """Calculate the distance of two point-clouds by ICP.

    Args:
        pc (ndarray): the point-cloud that will be transformed
        ref_pc (ndarray): the reference point-cloud
        pre_transform (ndarray): 4D affine matrix applied to pc before ICP. Defaults to None.
        icp_object (pypointmatcher icp object, optional): the ICP objects defining the pipeline.

    Returns:
        dict: {dist: mean distance oafter ICP, transformation: 4D affine transformation matrix}
    """
    dp = np2dp(pc)
    ref = np2dp(ref_pc)

    if pre_transform is not None:
        # TODO
        # pre transform the dp
        pass

    icp = icp_object

    # get the ICP object
    T = icp(dp, ref)  # calculate the ICP transformation

    data_out = DP(dp)  # copy DataPoint
    icp.transformations.apply(data_out, T)  # apply transformation to copy

    # initiate the matching with unfiltered point cloud
    icp.matcher.init(ref)
    # extract closest points
    matches = icp.matcher.findClosests(data_out)

    # weight paired points
    outlier_weights = icp.outlierFilters.compute(data_out, ref, matches)

    # generate tuples of matched points and remove pairs with zero weight
    matched_points = PM.ErrorMinimizer.ErrorElements(
        data_out, ref, outlier_weights, matches)

    # extract relevant information for convenience
    dim = matched_points.reading.getEuclideanDim()
    nb_matched_points = matched_points.reading.getNbPoints()
    matched_read = matched_points.reading.features[:dim]
    matched_ref = matched_points.reference.features[:dim]

    # compute mean distance
    dist = np.linalg.norm(matched_read - matched_ref, axis=0)
    mean_dist = dist.sum() / nb_matched_points

    return dict(
        dist=mean_dist,
        transformation=T  # a 4D affine matrix
    )


@_with_libpointmatcher
def align_with_icp(pc, ref_pc, icp_object=_DEFAULT_ICP):
    """Align pc to ref_pc by ICP.

    Args:
        pc (ndarray): point-cloud to align
        ref_pc (ndarray): reference point-cloud
        icp_object (pypointmatcher icp object, optional): the ICP objects defining the pipeline.

    Returns:
        ndarray : aligned point-cloud
    """
    dp = np2dp(pc)
    ref = np2dp(ref_pc)

    # get the ICP object
    icp = icp_object
    T = icp(dp, ref)  # get icp transformation

    dp_icp = DP(dp)

    # apply transformation
    icp.transformations.apply(dp_icp, T)
    return dp2np(dp_icp)
