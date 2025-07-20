from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# ------------------------------------------------------------------ 
# simple logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------


@dataclass
class ProjectInfo:
    name: str
    desc_path: Path          # description.txt
    project_dir: Path
    data_files: Dict[str, List[Path]] = field(default_factory=dict)


class DataAnalyzer:
    """
    PHIÊN BẢN RÚT GỌN của lớp bạn gửi – chỉ giữ những hàm
    cần thiết cho việc quét dữ liệu + gom paths.
    """
    SUPPORTED_DATA_EXTS = {".csv", ".parquet", ".tsv", ".txt",
                           ".json", ".jsonl", ".xlsx", ".xls"}
    SUPPORTED_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    SUPPORTED_TEXT_EXTS = {".txt", ".json", ".xml", ".html", ".md"}
    SUPPORTED_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".aac"}

    # ------------ public API ---------------------------------------
    def analyze(self, root: Path) -> List[ProjectInfo]:
        projects: List[ProjectInfo] = []
        for desc_file in root.rglob("description.txt"):
            project_dir = desc_file.parent
            name = project_dir.name
            data_files = self._collect_data_files(project_dir, desc_file)
            projects.append(
                ProjectInfo(name, desc_file, project_dir, data_files)
            )
            logger.debug("Detected %s with %d groups of files",
                         name, len(data_files))
        return projects

    # ------------ helpers ------------------------------------------
    def _collect_data_files(
        self, project_dir: Path, desc_path: Path
    ) -> Dict[str, List[Path]]:
        files: Dict[str, List[Path]] = {}

        def add(path: Path):
            key = path.stem if path.is_file() else path.name
            files.setdefault(key, []).append(path)

        # Tabular
        for ext in self.SUPPORTED_DATA_EXTS:
            for p in project_dir.rglob(f"*{ext}"):
                if p != desc_path:
                    add(p)

        # Other modalities (images / text / audio folders)
        SUPPORTED_EXTS = (
            self.SUPPORTED_IMG_EXTS
            | self.SUPPORTED_TEXT_EXTS
            | self.SUPPORTED_AUDIO_EXTS
        )
        for d in project_dir.rglob("*"):
            if d.is_dir() and any(
                f.suffix.lower() in SUPPORTED_EXTS for f in d.rglob("*") if f.is_file()
            ):
                add(d)

        return files