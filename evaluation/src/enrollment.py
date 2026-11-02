"""Enrollment Module for Beacon Auth Evaluation.

Manages enrollment of multiple identities, template generation via Beacon's
aggregation logic, and storage of evaluation profiles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from beacon.enrollment import aggregate, l2_normalize
from .beacon_adapter import BeaconAdapter, FaceInferenceResult
from .dataset_loader import LFWIdentity

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnrolledProfile:
    """An enrolled identity template in the evaluation framework."""
    identity_name: str
    anonymous_id: str
    template: np.ndarray
    enrollment_image_paths: List[Path]
    n_samples: int
    mean_sim_to_template: float
    min_sim_to_template: float


class EvaluationEnrollmentManager:
    """Manages enrollment across evaluation cohorts."""

    def __init__(self, adapter: BeaconAdapter, extractor_fn=None) -> None:
        self.adapter = adapter
        self.extractor_fn = extractor_fn or self.adapter.extract_face
        self._enrolled_profiles: Dict[str, EnrolledProfile] = {}

    def enroll_identity(
        self,
        identity: LFWIdentity,
        samples_per_identity: int = 1,
    ) -> Optional[EnrolledProfile]:
        """Enrolls an identity using their first N images as enrollment samples.
        
        Args:
            identity: LFWIdentity instance.
            samples_per_identity: Number of initial images to aggregate for template (default 1).
            
        Returns:
            EnrolledProfile if enrollment succeeds (face detected), else None.
        """
        if identity.image_count == 0:
            log.warning("Cannot enroll %s: No images available.", identity.name)
            return None

        n_enroll = min(samples_per_identity, identity.image_count)
        enroll_paths = identity.image_paths[:n_enroll]

        embeddings: List[np.ndarray] = []
        for p in enroll_paths:
            res: FaceInferenceResult = self.extractor_fn(p)
            if res.success and res.embedding is not None:
                embeddings.append(res.embedding)
            else:
                log.warning("Enrollment frame detection failed for %s (%s): %s",
                            identity.name, p.name, res.error_message)

        if not embeddings:
            log.error("Failed to enroll %s: No usable face embeddings extracted.", identity.name)
            return None

        if len(embeddings) == 1:
            template = l2_normalize(embeddings[0])
            profile = EnrolledProfile(
                identity_name=identity.name,
                anonymous_id=identity.identity_id,
                template=template,
                enrollment_image_paths=enroll_paths,
                n_samples=1,
                mean_sim_to_template=1.0,
                min_sim_to_template=1.0,
            )
        else:
            stats = aggregate(embeddings)
            profile = EnrolledProfile(
                identity_name=identity.name,
                anonymous_id=identity.identity_id,
                template=stats.template,
                enrollment_image_paths=enroll_paths,
                n_samples=stats.n,
                mean_sim_to_template=stats.mean_sim,
                min_sim_to_template=stats.min_sim,
            )

        self._enrolled_profiles[identity.name] = profile
        return profile

    def enroll_cohort(
        self,
        identities: List[LFWIdentity],
        samples_per_identity: int = 1,
    ) -> Dict[str, EnrolledProfile]:
        """Enrolls all identities in a cohort."""
        profiles: Dict[str, EnrolledProfile] = {}
        for ident in identities:
            prof = self.enroll_identity(ident, samples_per_identity=samples_per_identity)
            if prof is not None:
                profiles[ident.name] = prof
        log.info("Successfully enrolled %d/%d identities.", len(profiles), len(identities))
        return profiles

