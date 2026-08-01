from .permutation import (
    compute_greedy_tsp_1d,
    compute_pca_permutation,
    compute_2d_permutations,
    compute_joint_bipartite_2d_permutation,
    compute_3d_tensor_permutations,
)
from .transforms import (
    get_dct_matrix_1d,
    get_walsh_matrix_1d,
    haar_dwt_2d,
    haar_idwt_2d,
    dct_3d_project,
)
from .layers import SpecRAMALinear
from .shared_layers import SharedSpecRAMAModel, SharedSpecRAMALinear
from .layers_3d import SpecRAMA3DLinearGroup
from .quantization import HierarchicalSpectralQuantizer
from .utils import inject_spec_rama_in_model, merge_spec_rama_modules, count_trainable_parameters

__all__ = [
    "compute_greedy_tsp_1d",
    "compute_pca_permutation",
    "compute_2d_permutations",
    "compute_joint_bipartite_2d_permutation",
    "compute_3d_tensor_permutations",
    "get_dct_matrix_1d",
    "get_walsh_matrix_1d",
    "haar_dwt_2d",
    "haar_idwt_2d",
    "dct_3d_project",
    "SpecRAMALinear",
    "SharedSpecRAMAModel",
    "SharedSpecRAMALinear",
    "SpecRAMA3DLinearGroup",
    "HierarchicalSpectralQuantizer",
    "inject_spec_rama_in_model",
    "merge_spec_rama_modules",
    "count_trainable_parameters",
]
