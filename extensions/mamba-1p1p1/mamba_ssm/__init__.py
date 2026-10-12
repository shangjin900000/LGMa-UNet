__version__ = "1.1.1"

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from mamba_ssm.modules.mamba_simple import Mamba

# ODGNet / point-cloud use only needs Mamba; LM head pulls transformers generation APIs.
try:
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
except ImportError:
    MambaLMHeadModel = None  # type: ignore[misc, assignment]
