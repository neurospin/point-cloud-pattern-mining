try:
    from pypointmatcher import pointmatcher as pm, pointmatchersupport as pms
    HAS_LIBPOINTMATCHER = True
except ImportError as e:
    HAS_LIBPOINTMATCHER = False

import functools
from selectors import EpollSelector
import numpy as np


_DEFAULT_ICP_OBJ = None


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
def new_ICP_object(epsilon=0.1, max_iter=10):
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
    params["epsilon"] = "0"
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
    trCheckMaxIter = PM.get().TransformationCheckerRegistrar.create(name, params)
    params.clear()
    icp.transformationCheckers.append(trCheckMaxIter)

    name = "DifferentialTransformationChecker"
    params["minDiffRotErr"] = str(epsilon)
    params["minDiffTransErr"] = str(epsilon)
    params["smoothLength"] = "4"
    diff = PM.get().TransformationCheckerRegistrar.create(name, params)
    params.clear()
    icp.transformationCheckers.append(diff)

    return icp


def string_of_icp_object(icp_object="default"):
    """Return a string describing the icp_object properties."""
    desc = list()
    desc.append(f"Error minimizer: {icp_object.errorMinimizer.className}")
    desc.append(f"Inspector: {icp_object.inspector.className}")
    desc.append(f"Matcher: {icp_object.matcher.className}")
    desc.append("")

    desc.append("REFERENCE FITLERS ===")
    if len(icp_object.referenceDataPointsFilters) == 0:
        desc.append("None")
    for prop in icp_object.referenceDataPointsFilters:
        desc.append(f"{prop.className}: {dict(prop.parameters.items())}")
    desc.append("")

    desc.append("DATA FILTERS ===")
    if len(icp_object.readingDataPointsFilters) == 0:
        desc.append("None")
    for prop in icp_object.readingDataPointsFilters:
        desc.append(f"{prop.className}: {dict(prop.parameters.items())}")
    desc.append("")

    desc.append("TRANSFORMATION CHECKERS ===")
    if len(icp_object.transformationCheckers) == 0:
        desc.append("None")
    for prop in icp_object.transformationCheckers:
        desc.append(f"{prop.className}: {dict(prop.parameters.items())}")
    desc.append("")

    desc.append("OUTLIER FILTERS ===")
    if len(icp_object.outlierFilters) == 0:
        desc.append("None")
    for prop in icp_object.outlierFilters:
        desc.append(f"{prop.className}: {dict(prop.parameters.items())}")
    desc.append("")
    return "\n".join(desc)

# def print_default_ICP_object_properties():
#     pass


@_with_libpointmatcher
def new_ICP_object_from_yaml(config_file: str):
    """Generate a custom ICP object using the libPointMatcher library, with the specified configuration file.
    The configuration file is a yaml formatted text file.
    See https://libpointmatcher.readthedocs.io/en/latest/Configuration/ for help

    Args:
        epsilon (float, optional): min distance improvement on one iteration.
        max_iter (int, optional): stop at this iteration.

    Returns:
        pypointmatcher ICP object
    """
    icp = PM.ICP()
    icp.loadFromYaml(config_file)
    return icp


@_with_libpointmatcher
def set_default_ICP_object_from_yaml(config_file: str):
    """Set the default ICP object using the specified configuration file.
    The configuration file is a yaml formatted text file.
    See https://libpointmatcher.readthedocs.io/en/latest/Configuration/ for help

    Args:
        config_file (str) : path to the YAML config file.
    """

    icp = PM.ICP()
    icp.loadFromYaml(config_file)
    set_default_icp_object(icp)


@_with_libpointmatcher
def reset_default_ICP_object():
    """Reset the default ICP object to module default."""
    set_default_icp_object(new_ICP_object())


@_with_libpointmatcher
def set_icp_object_parameters(epsilon: float, max_iter: int, icp_object="default"):
    """set epsilon and max_iter of the given icp object.

    Args:
        epsilon (float, optional): min distance improvement on one iteration.
        max_iter (int, optional): stop at this iteration.
        ICP_object (libpointmatcher icp object) the object representing the pipeline. Defaults to the modul's default.
    """

    if icp_object == "default":
        icp_object = get_default_icp_object()

    params = Parameters()

    # set epsilon
    name = "KDTreeMatcher"
    params["knn"] = "1"
    params["epsilon"] = "0"
    kdtree = PM.get().MatcherRegistrar.create(name, params)
    params.clear()
    icp_object.matcher = kdtree

    icp_object.transformationCheckers.clear()

    # Prepare transformation checker filters
    name = "CounterTransformationChecker"
    params["maxIterationCount"] = str(max_iter)
    trCheckMaxIter = PM.get().TransformationCheckerRegistrar.create(name, params)
    params.clear()
    icp_object.transformationCheckers.append(trCheckMaxIter)

    name = "DifferentialTransformationChecker"
    params["minDiffRotErr"] = str(epsilon)
    params["minDiffTransErr"] = str(epsilon)
    params["smoothLength"] = "4"
    diff = PM.get().TransformationCheckerRegistrar.create(name, params)
    params.clear()
    icp_object.transformationCheckers.append(diff)


@_with_libpointmatcher
def set_default_icp_object(icp_object):
    """Set the default icp object"""
    # TODO: assert type
    global _DEFAULT_ICP_OBJ
    _DEFAULT_ICP_OBJ = icp_object


def get_default_icp_object():
    """Get the default ICP object (lazy)."""
    global _DEFAULT_ICP_OBJ
    if HAS_LIBPOINTMATCHER and (_DEFAULT_ICP_OBJ is None):
        _DEFAULT_ICP_OBJ = new_ICP_object()

    return _DEFAULT_ICP_OBJ


@_with_libpointmatcher
def icp_libpointmatcher(moving: np.ndarray, model: np.ndarray, icp_object="default", epsilon=None, max_iter=None):
    """Calculate Iterative Closest Point (ICP) distance.

    Args:
        moving (np.ndarray): point-cloud that will be moved
        model (np.ndarray): reference point-cloud
        icp_object (pypointmatcher icp object, optional): the ICP objects defining the pipeline.
        epsilon=None, max_iter=None

    Returns:
        tuple : distance, rotation_matrix, translation_vector
    """
    if icp_object == "default":
        icp_object = get_default_icp_object()

    set_icp_object_parameters(epsilon=epsilon, max_iter=max_iter)
    d = calc_distance(moving, model, icp_object=icp_object)
    T = d['transformation']
    return d['dist'], T[0:3, 0:3], T[0:3, 3]


@_with_libpointmatcher
def calc_distance(pc, ref_pc, pre_transform=None, icp_object="default"):
    """Calculate the distance of two point-clouds by ICP.

    Args:
        pc (ndarray): the point-cloud that will be transformed
        ref_pc (ndarray): the reference point-cloud
        pre_transform (ndarray): 4D affine matrix applied to pc before ICP. Defaults to None.
        icp_object (pypointmatcher icp object, optional): the ICP objects defining the pipeline.

    Returns:
        dict: {dist: mean distance oafter ICP, transformation: 4D affine transformation matrix}
    """

    if icp_object == "default":
        icp_object = get_default_icp_object()

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
def align_with_icp(pc, ref_pc, icp_object="default"):
    """Align pc to ref_pc by ICP.

    Args:
        pc (ndarray): point-cloud to align
        ref_pc (ndarray): reference point-cloud
        icp_object (pypointmatcher icp object, optional): the ICP objects defining the pipeline.

    Returns:
        ndarray : aligned point-cloud
    """

    if icp_object == "default":
        icp_object = get_default_icp_object()

    dp = np2dp(pc)
    ref = np2dp(ref_pc)

    # get the ICP object
    icp = icp_object
    T = icp(dp, ref)  # get icp transformation

    dp_icp = DP(dp)

    # apply transformation
    icp.transformations.apply(dp_icp, T)
    return dp2np(dp_icp)
