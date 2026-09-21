import logging
from dataclasses import dataclass
from threading import Lock
from typing import List

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


@dataclass
class CandidateInput:
    id: str | int
    text: str


@dataclass
class ScoredCandidate:
    id: str | int
    score: float


class SbertSimilarityService:
    """Service for semantic similarity candidate ranking using SBERT."""

    def __init__(self, model_name: str = MODEL_NAME):
        try:
            logger.info("Loading SBERT model %s...", model_name)
            self.model = SentenceTransformer(model_name)

            actual_dim = self.model.get_sentence_embedding_dimension()
            if actual_dim != EMBEDDING_DIM:
                logger.warning(
                    "Expected embedding dimension %d, but got %d",
                    EMBEDDING_DIM, actual_dim
                )

            logger.info("SBERT model loaded successfully.")
        except Exception:
            logger.exception("Failed to load SBERT model %s.", model_name)
            raise

    def rank_candidates(
        self,
        query_text: str,
        candidates: List[CandidateInput],
        max_results: int = 5
    ) -> List[ScoredCandidate]:
        """
        Rank candidate texts against the query text based on cosine similarity.

        Args:
            query_text: The new complaint description.
            candidates: A list of candidate inputs (id and text).
            max_results: The maximum number of candidates to return.

        Returns:
            A ranked list of ScoredCandidate objects in descending order of similarity.
        """
        if not candidates:
            return []

        try:
            candidate_texts = [c.text for c in candidates]

            # Generate embeddings
            query_embedding = self.model.encode(query_text, convert_to_tensor=True)
            candidate_embeddings = self.model.encode(candidate_texts, convert_to_tensor=True)

            # Compute cosine similarity
            cos_scores = F.cosine_similarity(query_embedding.unsqueeze(0), candidate_embeddings)

            scores = cos_scores.cpu().tolist()

            scored_candidates = [
                ScoredCandidate(id=candidates[i].id, score=float(scores[i]))
                for i in range(len(candidates))
            ]

            # Sort descending by score
            scored_candidates.sort(key=lambda x: x.score, reverse=True)

            # Enforce max limit
            return scored_candidates[:max_results]

        except Exception:
            logger.exception("Failed to compute semantic similarity or rank candidates.")
            raise


_service_instance: SbertSimilarityService | None = None
_service_lock = Lock()


def get_similarity_service() -> SbertSimilarityService:
    """
    Return the process-level SBERT similarity service.
    The model is loaded lazily on the first call to this function.
    """
    global _service_instance

    if _service_instance is not None:
        return _service_instance

    with _service_lock:
        if _service_instance is not None:
            return _service_instance

        _service_instance = SbertSimilarityService()
        return _service_instance
