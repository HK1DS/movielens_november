import unittest

import numpy as np
import torch

from eval_slices import _blend_rerank_scores
from ikgr_core.corona_retriever import CoronaRetriever


class CoronaRerankTest(unittest.TestCase):
    def test_zero_lambda_keeps_base_scores(self):
        scores = torch.tensor([[0.3, float("-inf"), -0.2]])
        prior = torch.tensor([[float("nan"), 1.0, 0.0]])
        out = _blend_rerank_scores(scores, prior, torch.ones((1, 1)), 0.0)
        self.assertIs(out, scores)
        self.assertTrue(torch.equal(out, scores))

    def test_soft_rerank_does_not_accept_empty_candidate_count(self):
        retriever = object.__new__(CoronaRetriever)
        retriever.n_items = 3
        with self.assertRaises(ValueError):
            retriever.candidates(np.array([1], dtype=np.int64), 0)

    def test_normalized_prior_is_finite(self):
        retriever = object.__new__(CoronaRetriever)
        retriever._scores = lambda _: np.array([[np.nan, np.inf, 0.0]], dtype=np.float32)
        prior = retriever.normalized_prior(np.array([1], dtype=np.int64))
        self.assertTrue(np.isfinite(prior).all())
        np.testing.assert_array_equal(prior, np.zeros((1, 3), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
