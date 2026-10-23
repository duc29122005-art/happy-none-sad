from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .config import MultiViewFaceConfig
from .types import StereoFacePair
from .utils import cosine_similarity


@dataclass
class Track:
    person_id: str
    embedding: Optional[np.ndarray]
    last_seen_frame: int
    hits: int = 1


class IdentityTracker:
    """Simple cross-camera identity tracker over time."""

    def __init__(self, config: MultiViewFaceConfig) -> None:
        self.config = config
        self.tracks: Dict[str, Track] = {}
        self.next_id = 1

    def _new_id(self) -> str:
        person_id = f"{self.next_id:03d}"
        self.next_id += 1
        return person_id

    def _pair_embedding(self, pair: StereoFacePair) -> Optional[np.ndarray]:
        embeddings = [e for e in (pair.left.embedding, pair.right.embedding) if e is not None]
        if not embeddings:
            return None
        min_len = min(e.reshape(-1).shape[0] for e in embeddings)
        avg = np.mean([e.reshape(-1)[:min_len] for e in embeddings], axis=0).astype(np.float32)
        norm = np.linalg.norm(avg)
        return avg / norm if norm > 1e-8 else avg

    def assign(self, pair: StereoFacePair) -> str:
        frame_index = min(pair.left.frame_index, pair.right.frame_index)
        current = self._pair_embedding(pair)
        best_id = None
        best_score = -1.0
        for person_id, track in list(self.tracks.items()):
            if frame_index - track.last_seen_frame > self.config.max_track_age:
                continue
            score = cosine_similarity(current, track.embedding)
            if score > best_score:
                best_id = person_id
                best_score = score

        if best_id is None or best_score < self.config.track_similarity_threshold:
            best_id = self._new_id()
            self.tracks[best_id] = Track(best_id, current, frame_index)
        else:
            track = self.tracks[best_id]
            if current is not None:
                if track.embedding is None:
                    track.embedding = current
                else:
                    updated = 0.85 * track.embedding + 0.15 * current
                    norm = np.linalg.norm(updated)
                    track.embedding = updated / norm if norm > 1e-8 else updated
            track.last_seen_frame = frame_index
            track.hits += 1

        pair.person_id = best_id
        return best_id
