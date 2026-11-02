from __future__ import annotations

import torch  # noqa: F401 - Import before scitbx/omptbx_ext to avoid OpenMP runtime conflict

if torch.cuda.is_available() and torch.get_num_threads() > 1:
    torch.set_num_threads(1)

__version__ = "0.1.0"
