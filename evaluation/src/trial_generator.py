"""Trial Generator for Beacon Auth Evaluation.

Generates genuine, impostor, and blur stress-test trials with strict
data leakage prevention (enrollment and probe images strictly disjoint).
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

from .dataset_loader import LFWIdentity
from .enrollment import EnrolledProfile

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuthenticationTrial:
    """A single authentication trial definition."""
    trial_id: str
    experiment_id: str
    cohort: str
    enrolled_identity_name: str
    enrolled_identity_id: str
    probe_identity_name: str
    probe_identity_id: str
    probe_image_path: Path
    ground_truth: int  # 1 = Genuine, 0 = Impostor
    blur_sigma: float
    blur_kernel: int


class TrialGenerator:
    """Generates evaluation trials from enrolled profiles and identities."""

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed
        self._trial_counter = 1

    def _next_trial_id(self) -> str:
        tid = f"TR_{self._trial_counter:06d}"
        self._trial_counter += 1
        return tid

    def generate_genuine_trials(
        self,
        enrolled_profiles: Dict[str, EnrolledProfile],
        identities: Dict[str, LFWIdentity],
        cohort_name: str = "test",
        blur_sigma: float = 1.0,
        blur_kernel: int = 7,
        samples_per_enroll: int = 1,
    ) -> List[AuthenticationTrial]:
        """Generates genuine authentication trials for all available probe images."""
        trials: List[AuthenticationTrial] = []

        for name, profile in enrolled_profiles.items():
            ident = identities.get(name)
            if not ident:
                continue

            enroll_set: Set[Path] = {p.resolve() for p in profile.enrollment_image_paths}
            # All images after enrollment images are genuine probe candidates
            probe_paths = [p for p in ident.image_paths[samples_per_enroll:]]

            for probe_path in probe_paths:
                # Sanity check: Probe image MUST NOT be in enrollment set
                if probe_path.resolve() in enroll_set:
                    raise AssertionError(
                        f"Data leakage detected! Probe image {probe_path.name} was used in enrollment for {name}!"
                    )

                trial = AuthenticationTrial(
                    trial_id=self._next_trial_id(),
                    experiment_id="exp1_genuine",
                    cohort=cohort_name,
                    enrolled_identity_name=name,
                    enrolled_identity_id=profile.anonymous_id,
                    probe_identity_name=name,
                    probe_identity_id=ident.identity_id,
                    probe_image_path=probe_path,
                    ground_truth=1,
                    blur_sigma=blur_sigma,
                    blur_kernel=blur_kernel,
                )
                trials.append(trial)

        log.info("Generated %d genuine trials for cohort '%s'", len(trials), cohort_name)
        return trials

    def generate_impostor_trials(
        self,
        enrolled_profiles: Dict[str, EnrolledProfile],
        identities: Dict[str, LFWIdentity],
        cohort_name: str = "test",
        impostors_per_user: int = 5,
        blur_sigma: float = 1.0,
        blur_kernel: int = 7,
    ) -> List[AuthenticationTrial]:
        """Generates impostor authentication trials across different identities."""
        trials: List[AuthenticationTrial] = []
        rng = random.Random(self.random_seed)
        all_enrolled_names = sorted(list(enrolled_profiles.keys()))

        for name, profile in enrolled_profiles.items():
            # Potential other identities
            other_names = [n for n in all_enrolled_names if n != name]
            if not other_names:
                continue

            # Sample other identities
            n_sample = min(impostors_per_user, len(other_names))
            selected_impostors = rng.sample(other_names, n_sample)

            for imp_name in selected_impostors:
                imp_ident = identities[imp_name]
                # Pick a probe image from the impostor (e.g. index 1 or 0)
                imp_probe_idx = 1 if imp_ident.image_count > 1 else 0
                imp_probe_path = imp_ident.image_paths[imp_probe_idx]

                # Sanity check: Impostor probe must not belong to enrolled identity
                if imp_name == name:
                    raise AssertionError("Impostor probe has same identity as enrolled user!")

                trial = AuthenticationTrial(
                    trial_id=self._next_trial_id(),
                    experiment_id="exp2_impostor",
                    cohort=cohort_name,
                    enrolled_identity_name=name,
                    enrolled_identity_id=profile.anonymous_id,
                    probe_identity_name=imp_name,
                    probe_identity_id=imp_ident.identity_id,
                    probe_image_path=imp_probe_path,
                    ground_truth=0,
                    blur_sigma=blur_sigma,
                    blur_kernel=blur_kernel,
                )
                trials.append(trial)

        log.info("Generated %d impostor trials for cohort '%s'", len(trials), cohort_name)
        return trials

    def generate_blur_robustness_trials(
        self,
        enrolled_profiles: Dict[str, EnrolledProfile],
        identities: Dict[str, LFWIdentity],
        blur_sigmas: List[float],
        cohort_name: str = "test",
    ) -> List[AuthenticationTrial]:
        """Generates genuine and impostor trials across multiple blur levels."""
        all_blur_trials: List[AuthenticationTrial] = []
        from .preprocessing import auto_kernel

        for sigma in blur_sigmas:
            kernel = 3 if sigma == 0.0 else auto_kernel(sigma)
            gen = self.generate_genuine_trials(
                enrolled_profiles, identities, cohort_name=cohort_name,
                blur_sigma=sigma, blur_kernel=kernel
            )
            # Tag with blur experiment ID
            gen_tagged = [
                AuthenticationTrial(
                    trial_id=t.trial_id,
                    experiment_id=f"exp4_blur_s{sigma}",
                    cohort=t.cohort,
                    enrolled_identity_name=t.enrolled_identity_name,
                    enrolled_identity_id=t.enrolled_identity_id,
                    probe_identity_name=t.probe_identity_name,
                    probe_identity_id=t.probe_identity_id,
                    probe_image_path=t.probe_image_path,
                    ground_truth=t.ground_truth,
                    blur_sigma=t.blur_sigma,
                    blur_kernel=t.blur_kernel,
                )
                for t in gen
            ]

            imp = self.generate_impostor_trials(
                enrolled_profiles, identities, cohort_name=cohort_name,
                impostors_per_user=3, blur_sigma=sigma, blur_kernel=kernel
            )
            imp_tagged = [
                AuthenticationTrial(
                    trial_id=t.trial_id,
                    experiment_id=f"exp4_blur_s{sigma}",
                    cohort=t.cohort,
                    enrolled_identity_name=t.enrolled_identity_name,
                    enrolled_identity_id=t.enrolled_identity_id,
                    probe_identity_name=t.probe_identity_name,
                    probe_identity_id=t.probe_identity_id,
                    probe_image_path=t.probe_image_path,
                    ground_truth=t.ground_truth,
                    blur_sigma=t.blur_sigma,
                    blur_kernel=t.blur_kernel,
                )
                for t in imp
            ]
            all_blur_trials.extend(gen_tagged)
            all_blur_trials.extend(imp_tagged)

        log.info("Generated %d total blur robustness trials across %d blur levels.",
                 len(all_blur_trials), len(blur_sigmas))
        return all_blur_trials

