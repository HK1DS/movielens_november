#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MovieLens adapter -- SKELETON. Fill in the two TODO methods; nothing else in the
pipeline needs to change (see data_adapters/base.py for the contract).

Dataset version (FIXED): MovieLens 25M -- the same version as the earlier MovieLens repo
(github.com/HK1DS/hk1ds_jwkim_movielens, prepare_movielens.py, raw dir data/movielens/ml-25m).
  download   https://files.grouplens.org/datasets/movielens/ml-25m.zip  -> unzip to raw/ml-25m/
  size       25,000,095 ratings | 162,541 users | 62,423 movies  (GroupLens README)
  files      ratings.csv        userId,movieId,rating,timestamp   (rating 0.5-5.0, timestamp = UTC epoch seconds)
             movies.csv         movieId,title,genres              (genres "Adventure|Animation|Children",
                                                                   or "(no genres listed)")
             tags.csv           userId,movieId,tag,timestamp      (user-applied, time-stamped!)
             links.csv          movieId,imdbId,tmdbId
             genome-scores.csv  movieId,tagId,relevance  /  genome-tags.csv  tagId,tag
  Do not swap in ml-32m / ml-latest / ml-1m: ids, sizes and time ranges differ, so results would
  no longer be comparable with the earlier repo.

What the earlier repo did with ml-25m (reference / sanity check, not yet decided here):
  - all ratings kept as implicit interactions (no rating threshold)
  - movies with "(no genres listed)" dropped together with their ratings
      -> 24,973,468 ratings | 162,541 users | 54,479 movies
  - item profile text = "title | genres"; genres = the only metadata relation ("categories")
  - k=30 core -> 133,684 users | 15,785 items | 24,046,865 interactions
  (its user profiles = earliest 80% of each user's history; this repo replaces that with the
   exact train-only profiles of build_trainonly_profiles.py, so profiles will NOT be identical)

Decisions to make before filling in (they change results, so record them):
  [ ] Scale. Full 25M at k=30 is ~134k users; stage B/C makes one LLM call per unique non-empty
      profile (step1 + step2 each; the earlier repo's TO_GLOBAL run: ~113k profiles, ~188M tokens), and
      LightGCN propagates the whole train graph every batch -> check `run_pipeline.py --steps 0ATPS`
      cost estimate and GPU time before committing; raise --k or subsample users if needed.
  [ ] Positive feedback. All ratings as implicit positives (earlier repo + Goodreads setup) or
      rating >= 4? -> `dataset.min_rating` in config.yaml (applied in prepare_interactions.py).
  [ ] "(no genres listed)" movies: drop (earlier repo) or keep without genre nodes.
  [ ] Item text. movies.csv only has title + genres, so LLM item intents from it will mostly
      restate the genres (= the genre metadata channel). Richer static text needs an external
      source (e.g. TMDB overview via links.csv) -- cache it to a file and list it in source_files().
  [ ] Tags. tags.csv / tag genome are aggregated over ALL time, i.e. they include validation/test
      period behaviour. Either leave them out of item metadata, or rebuild them from train-period
      rows only (TO: per user cut, TO_GLOBAL: global cut). Do not silently use them.
  [ ] Timestamps. MovieLens timestamps are rating-entry times (users bulk-rate old movies at
      sign-up). Ties are handled by ikgr_core/temporal_split.py; ts_clip is usually not needed.
"""
import os

from .base import DatasetAdapter, ItemMeta, MetaRelation, ProfileFacet  # noqa: F401


class MovieLensAdapter(DatasetAdapter):
    name = "movielens"
    recent_label = "Recently watched (newest first)"
    # TODO(decide): placeholders. Relation names must match the keys load_items() fills.
    profile_facets = [ProfileFacet(relation="genre", label="Frequent genres", top_n=10)]
    meta_relations = [MetaRelation(name="genre")]
    ts_clip = None

    def __init__(self, raw_dir="raw/ml-25m", **opts):
        self.raw_dir = raw_dir
        self.opts = opts

    def _path(self, name):
        return os.path.join(self.raw_dir, name)

    def iter_interactions(self):
        # TODO: yield (str(userId), str(movieId), float(rating), float(timestamp)) from
        #       raw/ml-25m/ratings.csv. Stream the file; 25M rows do not need pandas here.
        #       Apply the "(no genres listed)" decision here too if those movies are dropped.
        raise NotImplementedError("MovieLensAdapter.iter_interactions: see the TODO in data_adapters/movielens.py")

    def load_items(self, item_ids):
        # TODO: return {movieId: ItemMeta(title=..., profile=..., relations={"genre": [...], ...})}
        #       for movieId in item_ids. title = display string for user profiles (e.g. "Toy Story (1995)"),
        #       profile = static item text for LLM item intents, relations = metadata-KG nodes.
        raise NotImplementedError("MovieLensAdapter.load_items: see the TODO in data_adapters/movielens.py")

    def source_files(self):
        # Files whose content feeds profiles / the metadata KG (hashed into the stage-D signature).
        names = ("movies.csv",)
        return [self._path(n) for n in names if os.path.exists(self._path(n))]
