#  Copyright 2022-2024 MTS (Mobile Telesystems)
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

"""Popular model."""

import typing as tp
from datetime import datetime, timedelta
from enum import Enum

import numpy as np
import pandas as pd
import typing_extensions as tpe
from pydantic import BeforeValidator, PlainSerializer
from tqdm.auto import tqdm

from rectools import Columns, InternalIds
from rectools.dataset import Dataset
from rectools.models.base import ModelConfig
from rectools.types import InternalIdsArray
from rectools.utils import fast_isin_for_sorted_test_elements

from .base import FixedColdRecoModelMixin, ModelBase, Scores, ScoresArray
from .utils import get_viewed_item_ids


class Popularity(Enum):
    """Types of popularity"""

    N_USERS = "n_users"
    N_INTERACTIONS = "n_interactions"
    MEAN_WEIGHT = "mean_weight"
    SUM_WEIGHT = "sum_weight"


def _deserialize_timedelta(td: tp.Any) -> tp.Any:
    if isinstance(td, dict):
        return timedelta(**td)
    return td


def _serialize_timedelta(td: timedelta) -> dict:
    serialized_td = {
        key: value
        for key, value in {"days": td.days, "seconds": td.seconds, "microseconds": td.microseconds}.items()
        if value != 0
    }
    return serialized_td


TimeDelta = tpe.Annotated[
    timedelta,
    BeforeValidator(func=_deserialize_timedelta),
    PlainSerializer(func=_serialize_timedelta, return_type=dict, when_used="json"),
]


class PopularModelConfig(ModelConfig):
    """Config for `PopularModel`."""

    popularity: Popularity = Popularity.N_USERS
    period: tp.Optional[TimeDelta] = None
    begin_from: tp.Optional[datetime] = None
    add_cold: bool = False
    inverse: bool = False


PopularityOptions = tp.Literal["n_users", "n_interactions", "mean_weight", "sum_weight"]


class PopularModelMixin:
    """Mixin for models based on popularity."""

    @classmethod
    def _validate_popularity(
        cls,
        popularity: PopularityOptions,
    ) -> Popularity:
        try:
            return Popularity(popularity)
        except ValueError:
            possible_values = {item.value for item in Popularity.__members__.values()}
            raise ValueError(f"`popularity` must be one of the {possible_values}. Got {popularity}.")

    @classmethod
    def _validate_time_attributes(
        cls,
        period: tp.Optional[TimeDelta],
        begin_from: tp.Optional[datetime],
    ) -> None:
        if period is not None and begin_from is not None:
            raise ValueError("Only one of `period` and `begin_from` can be set")

    @classmethod
    def _filter_interactions(
        cls, interactions: pd.DataFrame, period: tp.Optional[TimeDelta], begin_from: tp.Optional[datetime]
    ) -> pd.DataFrame:
        if begin_from is not None:
            interactions = interactions.loc[interactions[Columns.Datetime] >= begin_from]
        elif period is not None:
            begin_from = interactions[Columns.Datetime].max() - period
            interactions = interactions.loc[interactions[Columns.Datetime] >= begin_from]
        return interactions

    @classmethod
    def _get_groupby_col_and_agg_func(cls, popularity: Popularity) -> tp.Tuple[str, str]:
        if popularity == Popularity.N_USERS:
            return Columns.User, "nunique"
        if popularity == Popularity.N_INTERACTIONS:
            return Columns.User, "count"
        if popularity == Popularity.MEAN_WEIGHT:
            return Columns.Weight, "mean"
        if popularity == Popularity.SUM_WEIGHT:
            return Columns.Weight, "sum"
        raise ValueError(f"Unexpected popularity {popularity}")


class PopularModel(FixedColdRecoModelMixin, PopularModelMixin, ModelBase[PopularModelConfig]):
    """
    Model generating recommendations based on popularity of items.

    Parameters
    ----------
    popularity : {"n_users", "n_interactions", "mean_weight", "sum_weight"}, default `"n_users"`
        Method of calculating item popularity.
        To evaluate `popularity score` the following methods are available:
        - `n_users` - number of unique users that interacted with item;
        - `n_interactions` - number of interactions with item;
        - `mean_weight` - mean item interactions weight;
        - `sum_weight` - total item interactions weight.
    period : timedelta, optional, default ``None``
        Period before last interaction to consider interactions for popularity calculation.
        Either `period` or `begin_from` can be set at once.
        If both are ``None`` all interactions will be used.
    begin_from : datetime, optional, default ``None``
        Exact datetime to consider interactions from for popularity calculation.
        Either `period` or `begin_from` can be set at once.
        If both are ``None`` all interactions will be used.
    add_cold : bool, default ``False``
        If ``True`` cold items will be added to the end of popularity list and can be recommended.
        Item is cold if it's not present in interactions at all (but present in id map)
        or not present in last interactions defined by either `period` or `begin_from` arguments.
        Order of cold items is unpredictable.
        Cold items score will be equal to ``0``.
    inverse : bool, default ``False``
        If ``True`` least popular items will be selected.
    verbose : int, default ``0``
        Degree of verbose output. If ``0``, no output will be provided.
    """

    recommends_for_warm = False
    recommends_for_cold = True

    config_class = PopularModelConfig

    def __init__(
        self,
        popularity: PopularityOptions = "n_users",
        period: tp.Optional[timedelta] = None,
        begin_from: tp.Optional[datetime] = None,
        add_cold: bool = False,
        inverse: bool = False,
        verbose: int = 0,
    ):
        super().__init__(
            verbose=verbose,
        )
        self.popularity = self._validate_popularity(popularity)
        self._validate_time_attributes(period, begin_from)
        self.period = period
        self.begin_from = begin_from

        self.add_cold = add_cold
        self.inverse = inverse

        self.popularity_list: tp.Tuple[InternalIdsArray, ScoresArray]

    def _get_config(self) -> PopularModelConfig:
        return PopularModelConfig(
            cls=self.__class__,
            popularity=self.popularity,
            period=self.period,
            begin_from=self.begin_from,
            add_cold=self.add_cold,
            inverse=self.inverse,
            verbose=self.verbose,
        )

    @classmethod
    def _from_config(cls, config: PopularModelConfig) -> tpe.Self:
        return cls(
            popularity=config.popularity.value,
            period=config.period,
            begin_from=config.begin_from,
            add_cold=config.add_cold,
            inverse=config.inverse,
            verbose=config.verbose,
        )

    def _fit(self, dataset: Dataset) -> None:  # type: ignore
        interactions = self._filter_interactions(dataset.interactions.df, self.period, self.begin_from)

        col, func = self._get_groupby_col_and_agg_func(self.popularity)
        items_scores = interactions.groupby(Columns.Item)[col].agg(func).sort_values(ascending=False)
        items = items_scores.index.values
        scores = items_scores.values.astype(float)

        if self.add_cold:
            cold_items = np.setdiff1d(dataset.item_id_map.internal_ids, items)
            items = np.concatenate((items, cold_items))
            scores = np.concatenate((scores, np.zeros(cold_items.size)))

        if self.inverse:
            items = items[::-1]
            scores = scores[::-1]

        self.popularity_list = (items, scores)

    def _recommend_u2i(
        self,
        user_ids: InternalIdsArray,
        dataset: Dataset,
        k: int,
        filter_viewed: bool,
        sorted_item_ids_to_recommend: tp.Optional[InternalIdsArray],
    ) -> tp.Tuple[InternalIds, InternalIds, Scores]:
        popularity_list = self._get_filtered_popularity_list(sorted_item_ids_to_recommend)

        if filter_viewed:
            user_items = dataset.get_user_item_matrix(include_weights=False)

        all_user_ids = []
        all_reco_ids: tp.List[int] = []
        all_scores: tp.List[float] = []
        for user_id in tqdm(user_ids, disable=self.verbose == 0):
            if filter_viewed:
                sorted_blacklist = get_viewed_item_ids(user_items, user_id)
            else:
                sorted_blacklist = None
            reco_ids, reco_scores = self._recommend_for_user(k, popularity_list, sorted_blacklist)
            all_user_ids.extend([user_id] * len(reco_ids))
            all_reco_ids.extend(reco_ids)
            all_scores.extend(reco_scores)

        return all_user_ids, all_reco_ids, all_scores

    @classmethod
    def _recommend_for_user(
        cls,
        k: int,
        popularity_list: tp.Tuple[InternalIdsArray, ScoresArray],
        sorted_blacklist: tp.Optional[InternalIdsArray],
    ) -> tp.Tuple[InternalIds, Scores]:
        if sorted_blacklist is not None:
            n_items = k + sorted_blacklist.size
        else:
            n_items = k

        reco = popularity_list[0][:n_items]
        scores = popularity_list[1][:n_items]

        if sorted_blacklist is not None:
            valid_mask = fast_isin_for_sorted_test_elements(reco, sorted_blacklist, invert=True)
            reco = reco[valid_mask][:k]
            scores = scores[valid_mask][:k]

        return reco, scores

    def _recommend_i2i(
        self,
        target_ids: InternalIdsArray,
        dataset: Dataset,
        k: int,
        sorted_item_ids_to_recommend: tp.Optional[InternalIdsArray],
    ) -> tp.Tuple[InternalIds, InternalIds, Scores]:
        _, single_reco, single_scores = self._recommend_u2i(
            user_ids=dataset.user_id_map.internal_ids[:1],
            dataset=dataset,
            k=k,
            filter_viewed=False,
            sorted_item_ids_to_recommend=sorted_item_ids_to_recommend,
        )

        n_targets = len(target_ids)
        n_reco_per_target = len(single_reco)

        all_target_ids = np.repeat(target_ids, n_reco_per_target)
        all_reco_ids = np.tile(single_reco, n_targets)
        all_scores = np.tile(single_scores, n_targets)
        return all_target_ids, all_reco_ids, all_scores

    def _get_filtered_popularity_list(
        self, sorted_item_ids_to_recommend: tp.Optional[InternalIdsArray]
    ) -> tp.Tuple[InternalIdsArray, ScoresArray]:
        popularity_list = self.popularity_list
        if sorted_item_ids_to_recommend is not None:
            valid_items_mask = fast_isin_for_sorted_test_elements(popularity_list[0], sorted_item_ids_to_recommend)
            popularity_list = (popularity_list[0][valid_items_mask], popularity_list[1][valid_items_mask])
        return popularity_list

    def _get_cold_reco(
        self, dataset: Dataset, k: int, sorted_item_ids_to_recommend: tp.Optional[InternalIdsArray]
    ) -> tp.Tuple[InternalIds, Scores]:
        popularity_list = self._get_filtered_popularity_list(sorted_item_ids_to_recommend)
        reco_ids = popularity_list[0][:k]
        scores = popularity_list[1][:k]
        return reco_ids, scores
