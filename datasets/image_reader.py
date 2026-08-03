import os

import cv2


import tarfile
import io
from pathlib import Path
from typing import Optional, Dict
import numpy as np
from PIL import Image


class TarImageReader:
    """Image reader that can read from both filesystem and tar archives."""

    def __init__(self, tar_path: Optional[str] = None):
        """
        Args:
            tar_path: Path to tar archive. If None, reads from filesystem.
        """
        self.tar_path = tar_path
        self.tar_file = None
        self.member_index: Dict[str, tarfile.TarInfo] = {}

        self.tar_file = tarfile.open(tar_path, 'r')
        # Build index for O(1) lookup
        for member in self.tar_file.getmembers():
            if member.isfile():
                self.member_index[member.name] = member
                # print("added ", member)
        print(f"Loaded tar with {len(self.member_index)} files")

    def __call__(self, path: str, is_mask: bool = False) -> np.ndarray:
        """
        Read image from tar archive or filesystem.

        Args:
            path: Path to image (relative to tar root or absolute filesystem path)
            is_mask: Whether this is a mask image

        Returns:
            numpy array of image
        """
        return self._read_from_tar(path, is_mask)

    def _read_from_tar(self, path: str, is_mask: bool) -> np.ndarray:
        """Read image from tar archive."""
        # Normalize path (remove leading slash if present)
        path = path.lstrip('/')

        # Try to find the file in tar
        if path not in self.member_index:
            # Try without prefix (in case meta file has full paths)
            path_parts = Path(path).parts
            for i in range(len(path_parts)):
                candidate = str(Path(*path_parts[i:]))
                if candidate in self.member_index:
                    path = candidate
                    break
            else:
                raise FileNotFoundError(f"Path not found in tar: {path}")

        member = self.member_index[path]
        # print(path, member)
        f = self.tar_file.extractfile(member)
        img_bytes = f.read()
        img = Image.open(io.BytesIO(img_bytes))

        if is_mask:
            img = img.convert('L')
        else:
            img = img.convert('RGB')

        return np.array(img)

    def __del__(self):
        if self.tar_file is not None:
            self.tar_file.close()


class OpenCVReader:
    def __init__(self, image_dir, color_mode):
        self.image_dir = image_dir
        self.color_mode = color_mode
        assert color_mode in ["RGB", "BGR",
                              "GRAY"], f"{color_mode} not supported"
        if color_mode != "BGR":
            self.cvt_color = getattr(cv2, f"COLOR_BGR2{color_mode}")
        else:
            self.cvt_color = None

    def __call__(self, filename, is_mask=False):
        filename = os.path.join(self.image_dir, filename)
        assert os.path.exists(filename), filename
        if is_mask:
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            return img
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        if self.color_mode != "BGR":
            img = cv2.cvtColor(img, self.cvt_color)
        return img


def build_image_reader(cfg_reader):
    if cfg_reader["type"] == "opencv":
        return OpenCVReader(**cfg_reader["kwargs"])
    elif cfg_reader["type"] == "tar":
        tar_path = cfg_reader.get('tar_path', None)
        if tar_path:
            return TarImageReader(tar_path)
        else:
            return OpenCVReader(**cfg_reader["kwargs"])

    else:
        raise TypeError(
            "no supported image reader type: {}".format(cfg_reader["type"]))
