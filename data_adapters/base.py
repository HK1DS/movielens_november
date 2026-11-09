#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset adapter contract.

Everything dataset-specific in the pipeline goes through ONE adapter object.
The rest of the code (temporal split replica, train-only profiles, LLM steps,
KG packs, IKGR model, evaluation, paired comparison) never reads raw files.

An adapter answers three questions:

  1. iter_interactions()  -> raw (user_id, item_id, rating, timestamp) tuples
                             (stage 0, prepare_interactions.py)
  2. load_items(item_ids) -> per-item metadata (ItemMeta):
        title      short display string used in USER profiles ("Recently ...: <title>; ...")
        profile    static item text for LLM ITEM intents (step1/step2)
        relations  {relation_name: [node keys]} for the heterogeneous metadata KG
                   (stage D, build_meta_kg.py) and for user-profile facets (stage P)
  3. how user profiles and the metadata KG are shaped:
        recent_label     header of the "recent items" part of a user profile
        profile_facets   which relations are aggregated into a user profile
        meta_relations   which relations become metadata-KG node types
        ts_clip          optional (lo, hi) clamp window for timestamps (YYYY-MM-DD)

Leakage rule for adapters: item metadata must be STATIC catalog information.
Anything aggregated from user activity over time (user-applied tags, review
text, "popular shelves" counts, ...) also contains validation/test-period
behaviour. If you use such a signal, document the decision or rebuild it from
train-period rows only.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple


@dataclass
class ItemMeta:
    title: str = ""
    profile: str = ""
    relations: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class ProfileFacet:
    """One aggregated line of a structured user profile, e.g. 'Frequent genres: ...'."""
    relation: str                 # key in ItemMeta.relations
    label: str                    # text printed before the values
    top_n: int = 10               # max values printed
    min_count: int = 1            # value must occur >= min_count times in the user's window
    primary_only: bool = False    # count only the first node of the relation per item
    show_counts: bool = False     # "value (3)" instead of "value"


@dataclass
class MetaRelation:
    """One node type of the heterogeneous metadata KG (item -- relation -- node)."""
    name: str                     # key in ItemMeta.relations
    max_per_item: Optional[int] = None   # keep the first N nodes per item (None = all)
    min_df: int = 1               # drop nodes linked to fewer than min_df items


class DatasetAdapter(ABC):
    name: str = "base"
    recent_label: str = "Recent items (newest first)"
    profile_facets: List[ProfileFacet] = []
    meta_relations: List[MetaRelation] = []
    ts_clip: Optional[Tuple[str, str]] = None

    @abstractmethod
    def iter_interactions(self) -> Iterable[Tuple[str, str, float, float]]:
        """Yield (user_id, item_id, rating, timestamp_epoch_seconds) in any order.
        Use 0.0 for a missing timestamp (make_temporal_inter.py clamps it)."""

    @abstractmethod
    def load_items(self, item_ids: Set[str]) -> Dict[str, ItemMeta]:
        """Return {item_id: ItemMeta} for the requested ids (missing ids may be absent)."""

    def source_files(self) -> List[str]:
        """Raw files whose CONTENT feeds profiles / metadata KG. Hashed into the
        stage-D signature so a changed catalog rebuilds the KG packs."""
        return []
