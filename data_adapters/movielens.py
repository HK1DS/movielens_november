#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MovieLens adapter (ml-25m). Contract: data_adapters/base.py.

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

Choices (2026-11-10: same as the earlier repo, for comparability; change via config options)
  - all ratings kept as implicit interactions      -> dataset.min_rating: null
  - "(no genres listed)" movies dropped together with their ratings
      -> 24,973,468 ratings | 162,541 users | 54,479 movies  (options.drop_no_genres: true)
  - item profile text = "<title> | <genres>", e.g. "Toy Story (1995) | Adventure|Animation|Children|Comedy|Fantasy"
  - genres = the only metadata-KG relation ("genre") and the only user-profile facet
  - user profile (2026-11-10, differs from the earlier repo): the user's 15 highest-rated titles from the
    WHOLE train-only history (ties: newest first) + genre counts over that whole history, e.g.
    "Favorite movies (highest rated first): Heat (1995); ... | Frequent genres: Drama (412), Comedy (300), ..."
  - tags.csv / tag genome NOT used (all-time aggregates -> validation/test leakage)
  Reference from the earlier repo: k=30 core -> 133,684 users | 15,785 items | 24,046,865 interactions.
  User profiles are this repo's exact train-only profiles, so their TEXT differs from the earlier
  repo's "earliest 80% of history" profiles (and so do the LLM intents).

Open points (do not change silently)
  - Item text is thin (title + genres): LLM item intents will largely restate the genres, i.e.
    overlap with the genre metadata channel. Richer static text needs an external source
    (e.g. TMDB overview via links.csv): cache it to a file and add it to source_files().
  - Timestamps are rating-entry times (users bulk-rate old movies at sign-up). Ties are handled
    by ikgr_core/temporal_split.py; ts_clip is not needed (all timestamps are valid epochs).
"""
import csv
import os

from .base import DatasetAdapter, ItemMeta, MetaRelation, ProfileFacet  # noqa: F401


NO_GENRES = "(no genres listed)"


def _clean(text):
    return " ".join(str(text or "").split())


class MovieLensAdapter(DatasetAdapter):
    name = "movielens"
    # MovieLens timestamps are rating-ENTRY times: many users rate hundreds of films in one sitting,
    # so their "newest 50" is an arbitrary (often alphabetical) batch. Explicit ratings are the
    # stronger taste signal, and recency is modelled separately (timestamps -> recency channel).
    # -> top-rated titles from the whole train history + genre distribution of that history.
    recent_label = "Favorite movies (highest rated first)"
    title_order = "rating"
    title_n = 15
    profile_window = None
    profile_facets = [ProfileFacet(relation="genre", label="Frequent genres", top_n=10, show_counts=True)]
    meta_relations = [MetaRelation(name="genre")]
    ts_clip = None

    def __init__(self, raw_dir="raw/ml-25m", drop_no_genres=True):
        self.raw_dir = raw_dir
        self.drop_no_genres = bool(drop_no_genres)
        self._movies = None

    def _path(self, name):
        return os.path.join(self.raw_dir, name)

    def _require(self, name):
        p = self._path(name)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found. Download https://files.grouplens.org/datasets/movielens/ml-25m.zip "
                f"and unzip it so that {self._path('ratings.csv')} exists "
                f"(scripts/run_intent_augmentation.ps1 does this on Windows).")
        return p

    def _load_movies(self):
        """{movieId: (title, [genres], raw genres string)} for every movie in movies.csv."""
        if self._movies is None:
            movies = {}
            with open(self._require("movies.csv"), encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    genres_raw = _clean(row.get("genres"))
                    genres = [] if genres_raw == NO_GENRES else [g for g in genres_raw.split("|") if g]
                    movies[str(row["movieId"])] = (_clean(row.get("title")), genres, genres_raw)
            self._movies = movies
        return self._movies

    def iter_interactions(self):
        movies = self._load_movies()
        skip = {m for m, (_, genres, _) in movies.items() if not genres} if self.drop_no_genres else set()
        with open(self._require("ratings.csv"), encoding="utf-8-sig", newline="") as f:
            r = csv.reader(f)
            header = next(r)
            iu, im, ir, it = (header.index(c) for c in ("userId", "movieId", "rating", "timestamp"))
            for row in r:
                if not row or row[im] in skip:
                    continue
                yield row[iu], row[im], float(row[ir]), float(row[it])

    def load_items(self, item_ids):
        movies = self._load_movies()
        out = {}
        for iid in item_ids:
            m = movies.get(str(iid))
            if m is None:
                continue
            title, genres, genres_raw = m
            profile = " | ".join(p for p in (title, genres_raw) if p)      # earlier repo's item text
            out[str(iid)] = ItemMeta(title=title, profile=profile, relations={"genre": genres})
        return out

    # Sizes of the earlier repo's runs on the same ml-25m files (HK1DS/hk1ds_jwkim_movielens README,
    # "Full MovieLens 25M k=30 final evaluation package"), valid for drop_no_genres=True and no
    # rating threshold. run_pipeline.py checks stage 0 / A against them.
    REFERENCE = {
        "prepared": {"interactions": 24_973_468, "users": 162_541, "items": 54_479},
        "kcore": {30: {"interactions": 24_046_865, "users": 133_684, "items": 15_785}},
    }

    def reference_stats(self, k):
        if not self.drop_no_genres or k not in self.REFERENCE["kcore"]:
            return None
        return {"prepared": self.REFERENCE["prepared"], "kcore": self.REFERENCE["kcore"][k],
                "source": "HK1DS/hk1ds_jwkim_movielens (ml-25m, k=30 final package)"}

    def source_files(self):
        # Files whose content feeds profiles / the metadata KG (hashed into the stage-D signature).
        return [self._path(n) for n in ("movies.csv",) if os.path.exists(self._path(n))]
