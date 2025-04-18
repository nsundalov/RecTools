#  Copyright 2022-2025 MTS (Mobile Telesystems)
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# pylint: disable=wrong-import-position

"""
Recommendation models (:mod:`rectools.models`)
==============================================

Convenient wrappers for popular recommendation
algorithms (ItemKNN, ALS, LightFM), also some custom
implementations.


Models
------
`models.DSSMModel`
`models.EASEModel`
`models.ImplicitALSWrapperModel`
`models.ImplicitBPRWrapperModel`
`models.ImplicitItemKNNWrapperModel`
`models.LightFMWrapperModel`
`models.PopularModel`
`models.PopularInCategoryModel`
`models.PureSVDModel`
`models.RandomModel`
`models.nn.bert4rec.BERT4RecModel`
`models.nn.sasrec.SASRecModel`
"""

from .ease import EASEModel
from .implicit_als import ImplicitALSWrapperModel
from .implicit_bpr import ImplicitBPRWrapperModel
from .implicit_knn import ImplicitItemKNNWrapperModel
from .popular import PopularModel
from .popular_in_category import PopularInCategoryModel
from .pure_svd import PureSVDModel
from .random import RandomModel
from .serialization import load_model, model_from_config, model_from_params

try:
    from .lightfm import LightFMWrapperModel
except ImportError:  # pragma: no cover
    from ..compat import LightFMWrapperModel  # type: ignore

try:
    from .nn.dssm import DSSMModel
    from .nn.transformers.bert4rec import BERT4RecModel
    from .nn.transformers.sasrec import SASRecModel
except ImportError:  # pragma: no cover
    from ..compat import BERT4RecModel, DSSMModel, SASRecModel  # type: ignore


__all__ = (
    "SASRecModel",
    "BERT4RecModel",
    "EASEModel",
    "ImplicitALSWrapperModel",
    "ImplicitBPRWrapperModel",
    "ImplicitItemKNNWrapperModel",
    "LightFMWrapperModel",
    "PopularModel",
    "PopularInCategoryModel",
    "PureSVDModel",
    "RandomModel",
    "DSSMModel",
    "load_model",
    "model_from_config",
    "model_from_params",
)
