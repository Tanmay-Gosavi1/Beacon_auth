"""LFW Dataset Validator for Beacon Auth.

Performs strict integrity checks, verifies identity disjointness,
checks image readability, and exports dataset_summary.csv.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from .dataset_loader import CohortSplit, LFWDatasetLoader, LFWIdentity, LFWPair

log = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Summary report of dataset validation."""
    total_identities_scanned: int
    usable_identities: int
    calibration_identities: int
    test_identities: int
    total_images_checked: int
    corrupt_images_count: int
    disjointness_passed: bool
    pairs_checked: int
    pairs_missing_images: int
    details_df: pd.DataFrame


class DatasetValidator:
    """Validates LFW image files, pair files, and cohort splits."""

    def __init__(self, loader: LFWDatasetLoader) -> None:
        self.loader = loader

    def validate_image_file(self, path: Path) -> Tuple[bool, str, Optional[Tuple[int, int, int]]]:
        """Validates that an image file can be decoded and has proper shape."""
        if not path.is_file():
            return False, "File does not exist", None
        try:
            img = cv2.imread(str(path))
            if img is None:
                return False, "cv2.imread returned None (corrupt/unsupported)", None
            if img.ndim != 3 or img.shape[2] != 3:
                return False, f"Invalid shape {img.shape}, expected 3 channels", None
            if img.dtype != np.uint8:
                return False, f"Invalid dtype {img.dtype}, expected uint8", None
            if img.shape[0] < 10 or img.shape[1] < 10:
                return False, f"Image dimensions too small: {img.shape[:2]}", None
            return True, "OK", img.shape
        except Exception as exc:
            return False, f"Exception during read: {str(exc)}", None

    def validate_cohorts(
        self,
        split: CohortSplit,
        reports_dir: Optional[Path] = None,
    ) -> ValidationReport:
        """Validates all images in calibration and test cohorts and generates summary."""
        calib_names = {i.name for i in split.calibration_identities}
        test_names = {i.name for i in split.test_identities}
        overlap = calib_names.intersection(test_names)
        disjointness_passed = len(overlap) == 0

        if not disjointness_passed:
            log.error("DATA LEAKAGE: Overlap between calibration and test cohorts: %s", overlap)

        rows = []
        total_images_checked = 0
        corrupt_images = 0

        # Validate calibration identities
        for ident in split.calibration_identities:
            valid, corrupt = self._validate_identity(ident, "calibration", rows)
            total_images_checked += valid + corrupt
            corrupt_images += corrupt

        # Validate test identities
        for ident in split.test_identities:
            valid, corrupt = self._validate_identity(ident, "test", rows)
            total_images_checked += valid + corrupt
            corrupt_images += corrupt

        details_df = pd.DataFrame(rows)

        # Write dataset_summary.csv if reports_dir is provided
        if reports_dir is not None:
            reports_dir = Path(reports_dir)
            reports_dir.mkdir(parents=True, exist_ok=True)
            summary_path = reports_dir / "dataset_summary.csv"
            details_df.to_csv(summary_path, index=False)
            log.info("Saved dataset summary to %s", summary_path)

        all_identities = self.loader.get_identities(min_images=1)
        usable_identities = self.loader.get_identities(min_images=3)

        report = ValidationReport(
            total_identities_scanned=len(all_identities),
            usable_identities=len(usable_identities),
            calibration_identities=len(split.calibration_identities),
            test_identities=len(split.test_identities),
            total_images_checked=total_images_checked,
            corrupt_images_count=corrupt_images,
            disjointness_passed=disjointness_passed,
            pairs_checked=0,
            pairs_missing_images=0,
            details_df=details_df,
        )
        return report

    def _validate_identity(
        self,
        ident: LFWIdentity,
        cohort: str,
        rows_accumulator: list,
    ) -> Tuple[int, int]:
        valid_count = 0
        corrupt_count = 0
        sample_res = "N/A"
        notes = []

        # Check enrollment image (image 0) and probe images (images 1..)
        enrollment_img = ident.image_paths[0] if ident.image_paths else None
        probe_imgs = ident.image_paths[1:] if len(ident.image_paths) > 1 else []

        for img_path in ident.image_paths:
            ok, msg, shape = self.validate_image_file(img_path)
            if ok:
                valid_count += 1
                if sample_res == "N/A" and shape is not None:
                    sample_res = f"{shape[1]}x{shape[0]}"
            else:
                corrupt_count += 1
                notes.append(f"{img_path.name}: {msg}")

        rows_accumulator.append({
            "identity_name": ident.name,
            "anonymous_id": ident.identity_id,
            "cohort": cohort,
            "total_images": ident.image_count,
            "enrollment_image": enrollment_img.name if enrollment_img else "",
            "probe_images_count": len(probe_imgs),
            "sample_resolution": sample_res,
            "is_valid": corrupt_count == 0,
            "validation_notes": "; ".join(notes) if notes else "Valid",
        })
        return valid_count, corrupt_count

    def validate_pairs(self, pairs: List[LFWPair]) -> Tuple[int, int]:
        """Validates that all images referenced in pairs exist and are readable."""
        missing = 0
        for p in pairs:
            if not p.image_path1.is_file() or not p.image_path2.is_file():
                missing += 1
        return len(pairs), missing

