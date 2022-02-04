from . import moving_averages_tools
from .moving_averages_tools import calc_MA_volumes_batch, calc_MA_volumes_with_alignment
from . import files_utils
from .distance import calc_all_icp, find_MAD_outliers, get_center_subject
from . import distance
from .isomap import Isomap_dist
from .transform import load_bucket, load_buckets, align_pc_pair, align_pcs
from . import plot

from .recipes import get_isomap, subset_of_distances, subset_of_pcs, load_icp_result, load_pcs_from_npz

import logging
# === Logging ===
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
_ch = logging.StreamHandler()
_ch.setFormatter(logging.Formatter('>>> %(levelname)s %(name)s - %(message)s'))
log.addHandler(_ch)
