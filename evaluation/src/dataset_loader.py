"""LFW Dataset Loader for Beacon Auth Performance Evaluation.

Discovers, parses, and structures LFW face images and verification pairs
with strict identity disjointness between calibration and evaluation cohorts.
"""

from __future__ import annotations

import csv
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LFWIdentity:
    """Represents an enrolled identity in LFW."""
    name: str
    identity_id: str  # Anonymized or normalized ID (e.g., ID_001)
    folder_path: Path
    image_paths: List[Path]

    @property
    def image_count(self) -> int:
        return len(self.image_paths)


@dataclass(frozen=True)
class LFWPair:
    """Represents a pair for the standard verification protocol."""
    pair_id: int
    name1: str
    image_path1: Path
    name2: str
    image_path2: Path
    is_same: bool
    fold: Optional[int] = None


@dataclass
class CohortSplit:
    """Identity-disjoint calibration and test cohorts."""
    calibration_identities: List[LFWIdentity]
    test_identities: List[LFWIdentity]
    remaining_identities: List[LFWIdentity] = field(default_factory=list)


class LFWDatasetLoader:
    """Loads and structures the Labeled Faces in the Wild (LFW) dataset."""

    def __init__(
        self,
        base_dir: str | Path,
        images_dir: Optional[str | Path] = None,
        pairs_file: Optional[str | Path] = None,
        random_seed: int = 42,
    ) -> None:
        self.base_dir = Path(base_dir)
        if images_dir is not None:
            self.images_dir = Path(images_dir)
        else:
            # Check standard LFW locations
            deepfunneled = self.base_dir / "lfw-deepfunneled"
            self.images_dir = deepfunneled if deepfunneled.exists() else self.base_dir

        self.pairs_file = Path(pairs_file) if pairs_file else (self.base_dir / "pairs.csv")
        self.random_seed = random_seed
        self._identities: Dict[str, LFWIdentity] = {}
        self._pairs: List[LFWPair] = []

    def scan_identities(self) -> Dict[str, LFWIdentity]:
        """Scans image directories and indexes all identities."""
        if not self.images_dir.exists():
            raise FileNotFoundError(f"LFW images directory not found at: {self.images_dir}")

        identities: Dict[str, LFWIdentity] = {}
        sorted_dirs = sorted([d for d in self.images_dir.iterdir() if d.is_dir()])
        
        idx = 1
        for d in sorted_dirs:
            # Collect all image files (jpg, jpeg, png)
            imgs = sorted([
                p for p in d.iterdir()
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ])
            if imgs:
                anon_id = f"ID_{idx:04d}"
                identities[d.name] = LFWIdentity(
                    name=d.name,
                    identity_id=anon_id,
                    folder_path=d,
                    image_paths=imgs,
                )
                idx += 1

        self._identities = identities
        log.info("Discovered %d total identities with %d images in %s",
                 len(identities), sum(i.image_count for i in identities.values()), self.images_dir)
        return self._identities

    def get_identities(self, min_images: int = 1) -> List[LFWIdentity]:
        """Returns identities filtered by minimum image count."""
        if not self._identities:
            self.scan_identities()
        return [ident for ident in self._identities.values() if ident.image_count >= min_images]

    def create_cohort_split(
        self,
        min_images: int = 3,
        n_calibration: int = 50,
        n_test: int = 50,
    ) -> CohortSplit:
        """Creates strictly identity-disjoint calibration and test cohorts.
        
        No person in the calibration cohort appears in the test cohort.
        """
        usable = self.get_identities(min_images=min_images)
        total_usable = len(usable)
        total_requested = n_calibration + n_test

        if total_usable < total_requested:
            raise ValueError(
                f"Requested {total_requested} identities ({n_calibration} calib + {n_test} test), "
                f"but only {total_usable} identities have >= {min_images} images."
            )

        rng = random.Random(self.random_seed)
        shuffled = list(usable)
        rng.shuffle(shuffled)

        calib_idents = sorted(shuffled[:n_calibration], key=lambda x: x.name)
        test_idents = sorted(shuffled[n_calibration:total_requested], key=lambda x: x.name)
        remaining = sorted(shuffled[total_requested:], key=lambda x: x.name)

        # Sanity check: Ensure strict disjointness
        calib_names = {i.name for i in calib_idents}
        test_names = {i.name for i in test_idents}
        overlap = calib_names.intersection(test_names)
        if overlap:
            raise AssertionError(f"Data leakage detected! Overlapping identities in cohorts: {overlap}")

        log.info("Created identity-disjoint cohorts: Calibration=%d, Test=%d, Remaining=%d",
                 len(calib_idents), len(test_idents), len(remaining))
        return CohortSplit(
            calibration_identities=calib_idents,
            test_identities=test_idents,
            remaining_identities=remaining,
        )

    def load_standard_pairs(self) -> List[LFWPair]:
        """Loads standard 10-fold LFW pairs from pairs.csv."""
        if not self.pairs_file.exists():
            log.warning("Pairs file not found at %s", self.pairs_file)
            return []

        pairs: List[LFWPair] = []
        with open(self.pairs_file, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            
            # Standard LFW pairs.csv has 6000 pairs: 10 folds of 300 genuine + 300 impostor
            pair_idx = 0
            for row in reader:
                if not row or not any(row):
                    continue
                pair_idx += 1
                fold_idx = (pair_idx - 1) // 600 + 1  # 1 to 10
                
                # Check row structure
                # Genuine: name, img1, img2, [empty]
                # Impostor: name1, img1, name2, img2
                if len(row) >= 4 and row[3].strip() != "":
                    # Impostor pair
                    name1 = row[0].strip()
                    num1 = int(float(row[1]))
                    name2 = row[2].strip()
                    num2 = int(float(row[3]))
                    is_same = False
                elif len(row) >= 3:
                    # Genuine pair
                    name1 = row[0].strip()
                    num1 = int(float(row[1]))
                    name2 = name1
                    num2 = int(float(row[2]))
                    is_same = True
                else:
                    continue

                path1 = self.images_dir / name1 / f"{name1}_{num1:04d}.jpg"
                path2 = self.images_dir / name2 / f"{name2}_{num2:04d}.jpg"

                pairs.append(LFWPair(
                    pair_id=pair_idx,
                    name1=name1,
                    image_path1=path1,
                    name2=name2,
                    image_path2=path2,
                    is_same=is_same,
                    fold=fold_idx,
                ))

        self._pairs = pairs
        log.info("Loaded %d standard LFW pairs across 10 folds", len(pairs))
        return self._pairs

    @staticmethod
    def load_image(path: Path | str) -> np.ndarray:
        """Loads an image from disk as BGR uint8 array."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to read image (corrupt or unreadable): {path}")
        return img

