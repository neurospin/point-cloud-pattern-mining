import numpy
from soma import aims

def aims_bucket_to_ndarray(aims_bucket: numpy.ndarray):
    """Transform an aims bucket into numpy array

    :param aims_bucket: aims bucket object
    :type aims_bucket: soma.aims.BucketMap_VOID.Bucket
    :return: a Nx3 array of points of the bucket
    :rtype: numpy.ndarray
    """
    assert isinstance(aims_bucket, aims.BucketMap_VOID.Bucket)

    v = numpy.empty((aims_bucket.size(), len(aims_bucket.keys()[0].list())))
    for i, point in enumerate(aims_bucket.keys()):
        v[i] = point.arraydata()
