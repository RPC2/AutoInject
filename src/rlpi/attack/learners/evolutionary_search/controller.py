"""
MAP Elites controller for evolutionary search attack.

Implements the island-based MAP Elites algorithm described in
Nasr et al. (2025) Appendix D. Candidates are stored in islands,
each with a grid indexed by (length_bin, diversity_bin). Each cell
holds at most one elite (the best-scoring candidate for that niche).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("MAPElitesController")


@dataclass
class Candidate:
    """A single candidate trigger with its evaluation metadata."""

    trigger: str
    score: float  # Numerical score (1-10)
    textual_feedback: str  # Qualitative feedback (critic reasoning)
    binary_success: bool  # Whether the attack succeeded
    length_bin: int = 0  # Quantized length category
    diversity_bin: int = 0  # Quantized diversity category
    iteration: int = 0  # When this candidate was created


@dataclass
class Island:
    """A single MAP Elites island with grid-based elite storage."""

    grid: Dict[Tuple[int, int], Candidate] = field(default_factory=dict)
    all_candidates: List[Candidate] = field(default_factory=list)

    def add_candidate(self, candidate: Candidate) -> bool:
        """Add candidate to island. Returns True if it became an elite."""
        self.all_candidates.append(candidate)
        cell_key = (candidate.length_bin, candidate.diversity_bin)

        current_elite = self.grid.get(cell_key)
        if current_elite is None or candidate.score > current_elite.score:
            self.grid[cell_key] = candidate
            return True
        return False

    def get_elites(self) -> List[Candidate]:
        """Return all elite candidates from the grid."""
        return list(self.grid.values())

    def sample_for_mutation(
        self, n: int, rng: np.random.RandomState
    ) -> List[Candidate]:
        """Sample a mix of elites and random candidates for mutation.

        Samples half from elites and half from all candidates (with
        replacement if needed).
        """
        elites = self.get_elites()
        if not elites and not self.all_candidates:
            return []

        samples: List[Candidate] = []

        # Sample from elites (up to half of n)
        n_elite = min(n // 2, len(elites))
        if n_elite > 0 and elites:
            elite_indices = rng.choice(len(elites), n_elite, replace=True)
            samples.extend(elites[i] for i in elite_indices)

        # Fill remainder from all candidates
        n_random = n - len(samples)
        if n_random > 0 and self.all_candidates:
            random_indices = rng.choice(
                len(self.all_candidates), n_random, replace=True
            )
            samples.extend(self.all_candidates[i] for i in random_indices)

        return samples


def _compute_length_bin(
    trigger: str, num_bins: int, bin_boundaries: List[int]
) -> int:
    """Compute length bin for a trigger based on character count."""
    length = len(trigger)
    for i, boundary in enumerate(bin_boundaries):
        if length < boundary:
            return i
    return num_bins - 1


def _compute_diversity_bin(
    trigger: str,
    existing_elites: List[Candidate],
    num_bins: int,
) -> int:
    """Compute diversity bin based on average normalized edit distance
    to existing elites.

    Uses a fast approximation: length-normalized Levenshtein distance
    against a sample of elites.
    """
    if not existing_elites:
        # First candidate gets the middle diversity bin
        return num_bins // 2

    # Sample up to 5 elites for efficiency
    sample = existing_elites[:5]
    distances = []
    for elite in sample:
        dist = _normalized_edit_distance(trigger, elite.trigger)
        distances.append(dist)

    avg_dist = np.mean(distances)

    # Quantize into bins (evenly spaced between 0 and 1)
    bin_width = 1.0 / num_bins
    return min(int(avg_dist / bin_width), num_bins - 1)


def _normalized_edit_distance(s1: str, s2: str) -> float:
    """Compute normalized Levenshtein edit distance between two strings.

    Returns value in [0, 1] where 0 = identical, 1 = completely different.
    Uses a space-optimized DP approach (two rows).
    """
    if s1 == s2:
        return 0.0

    len1, len2 = len(s1), len(s2)
    max_len = max(len1, len2)
    if max_len == 0:
        return 0.0

    # Space-optimized Levenshtein
    prev = list(range(len2 + 1))
    curr = [0] * (len2 + 1)

    for i in range(1, len1 + 1):
        curr[0] = i
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,  # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev, curr = curr, prev

    return prev[len2] / max_len


class MAPElitesController:
    """MAP Elites controller with multiple islands.

    Implements the controller component from Nasr et al. (2025):
    - Multiple islands for population diversity
    - Grid storage indexed by (length, diversity)
    - Sampling from elites + random candidates
    """

    def __init__(
        self,
        num_islands: int = 5,
        length_bins: int = 3,
        diversity_bins: int = 3,
        length_bin_boundaries: Optional[List[int]] = None,
        seed: int = 42,
    ):
        self.num_islands = num_islands
        self.length_bins = length_bins
        self.diversity_bins = diversity_bins
        self.rng = np.random.RandomState(seed)

        # Default boundaries: short(<100 chars), medium(<300), long(>=300)
        self.length_bin_boundaries = length_bin_boundaries or [100, 300]

        # Initialize islands
        self.islands = [Island() for _ in range(num_islands)]

        # Global tracking
        self.total_candidates = 0
        self.best_candidate: Optional[Candidate] = None

        logger.info(
            f"Initialized MAP Elites: {num_islands} islands, "
            f"{length_bins}x{diversity_bins} grid"
        )

    def add_candidate(self, candidate: Candidate) -> None:
        """Add a scored candidate to a random island."""
        # Assign to random island
        island_idx = self.rng.randint(0, self.num_islands)
        island = self.islands[island_idx]

        # Compute bins
        candidate.length_bin = _compute_length_bin(
            candidate.trigger, self.length_bins, self.length_bin_boundaries
        )
        candidate.diversity_bin = _compute_diversity_bin(
            candidate.trigger, island.get_elites(), self.diversity_bins
        )

        # Add to island
        became_elite = island.add_candidate(candidate)
        self.total_candidates += 1

        # Update global best
        if (
            self.best_candidate is None
            or candidate.score > self.best_candidate.score
        ):
            self.best_candidate = candidate

        if became_elite:
            logger.debug(
                f"New elite in island {island_idx} "
                f"cell ({candidate.length_bin}, {candidate.diversity_bin}): "
                f"score={candidate.score:.1f}"
            )

    def sample_parents(self, n_parents: int) -> List[Candidate]:
        """Sample parent candidates for mutation.

        Picks a random island and samples from it. If the island is
        empty, tries other islands.
        """
        # Shuffle island order
        island_order = self.rng.permutation(self.num_islands)

        for island_idx in island_order:
            island = self.islands[island_idx]
            if island.all_candidates:
                return island.sample_for_mutation(n_parents, self.rng)

        # No candidates anywhere
        return []

    def get_best_candidate(self) -> Optional[Candidate]:
        """Return the globally best candidate across all islands."""
        return self.best_candidate

    def get_all_elites(self) -> List[Candidate]:
        """Return all elite candidates across all islands."""
        elites = []
        for island in self.islands:
            elites.extend(island.get_elites())
        return elites

    def get_stats(self) -> Dict:
        """Return summary statistics."""
        total_elites = sum(len(island.grid) for island in self.islands)
        total_stored = sum(
            len(island.all_candidates) for island in self.islands
        )
        return {
            "total_candidates_evaluated": self.total_candidates,
            "total_elites": total_elites,
            "total_stored": total_stored,
            "best_score": (
                self.best_candidate.score if self.best_candidate else None
            ),
            "islands_with_candidates": sum(
                1 for island in self.islands if island.all_candidates
            ),
        }
