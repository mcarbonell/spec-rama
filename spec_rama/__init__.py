from .fourier_baseline import FourierFTLinear, inject_fourier_ft_in_model
from .layers import SpecRAMALinear
from .layers_3d import SpecRAMA3DLinearGroup
from .lora_baseline import LoRALinear, inject_lora_in_model
from .permutation import (
    compute_2d_permutations,
    compute_3d_tensor_permutations,
    compute_greedy_tsp_1d,
    compute_joint_bipartite_2d_permutation,
    compute_pca_permutation,
    list_permutation_methods,
    register_permutation_method,
)
from .quantization import HierarchicalSpectralQuantizer
from .shared_layers import SharedSpecRAMALinear, SharedSpecRAMAModel
from .transforms import (
    dct_3d_project,
    get_dct_matrix_1d,
    get_walsh_matrix_1d,
    haar_dwt_2d,
    haar_idwt_2d,
)
from .utils import (
    assert_strictly_frozen_base,
    count_trainable_parameters,
    inject_spec_rama_in_model,
    merge_spec_rama_modules,
    unmerge_spec_rama_modules,
)
from .vera_baseline import VeRALinear, VeRAModel

__all__ = [
    "compute_greedy_tsp_1d",
    "compute_pca_permutation",
    "compute_2d_permutations",
    "compute_joint_bipartite_2d_permutation",
    "compute_3d_tensor_permutations",
    "register_permutation_method",
    "list_permutation_methods",
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
    "unmerge_spec_rama_modules",
    "assert_strictly_frozen_base",
    "count_trainable_parameters",
    "LoRALinear",
    "inject_lora_in_model",
    "FourierFTLinear",
    "inject_fourier_ft_in_model",
    "VeRALinear",
    "VeRAModel",
]
