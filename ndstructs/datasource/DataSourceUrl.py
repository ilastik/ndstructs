from urllib.parse import urlparse
from typing import Union, List, Tuple
from pathlib import Path
import re
import os
from functools import partial
import h5py
import z5py


class DataSourceUrl:
    H5_EXTENSIONS = ["h5", "hdf5", "ilp"]
    N5_EXTENSIONS = ["n5"]
    ARCHIVE_TYPES_REGEX = "|".join(H5_EXTENSIONS + N5_EXTENSIONS)
    NOT_FOLLOWED_BY_DOUBLE_SLASH = r"(?!//)"

    @classmethod
    def is_remote(cls, path: Union[Path, str]) -> bool:
        return urlparse(Path(path).as_posix()).scheme not in ("file", "")

    @classmethod
    def split_archive_path(cls, path: Union[str, Path]) -> Tuple[Path, Path]:
        path = Path(path)
        if not cls.is_archive_path(path):
            raise ValueError(f"Path must be an archive path ({cls.ARCHIVE_TYPES_REGEX}). Provided: {path}")
        splitter = r"\.(" + cls.ARCHIVE_TYPES_REGEX + ")"
        external_path, extension, internal_path = re.split(splitter, path.absolute().as_posix(), re.IGNORECASE)
        return Path(external_path + "." + extension), Path(re.sub("^/", "", internal_path))

    @classmethod
    def is_archive_path(cls, path: Union[Path, str]) -> bool:
        # Any character after the trailing slash indicates something inside the archive
        ends_with_archive_extension = r"\.(" + cls.ARCHIVE_TYPES_REGEX + ")/."
        return bool(re.search(ends_with_archive_extension, Path(path).as_posix(), re.IGNORECASE))

    @classmethod
    def glob_to_regex(cls, glob: str) -> str:
        regex_star = "<REGEX_STAR>"
        return glob.replace("**", r"(?:\w|/)" + regex_star).replace("*", r"\w*").replace(regex_star, "*")

    @classmethod
    def sort_paths(cls, paths: List[Path]):
        return sorted(paths, key=lambda p: tuple(re.findall(r"[0-9]+", p.as_posix())))

    @classmethod
    def glob_archive_path(cls, path: Union[str, Path]) -> List[Path]:
        "Expands a path like /my/**/*/file.h5/some/**/*dataset* into all matching datasets in all matching files"

        external_path, internal_path = cls.split_archive_path(path)
        internal_regex = cls.glob_to_regex(internal_path.as_posix())
        dataset_paths: List[Path] = []

        def dataset_collector(object_path, obj, prefix: Path):
            if not isinstance(obj, (h5py._hl.dataset.Dataset, z5py.dataset.Dataset)):
                return
            if not re.match(internal_regex, object_path):
                return
            dataset_paths.append(prefix / object_path)

        for archive_path in cls.glob_fs_path(external_path):
            if archive_path.suffix.lower().replace(".", "") in cls.H5_EXTENSIONS:
                opener = partial(h5py.File, mode="r")
            else:
                opener = partial(z5py.File, mode="r")

            with opener(archive_path.as_posix()) as f:
                f.visititems(partial(dataset_collector, prefix=archive_path))

        return cls.sort_paths(dataset_paths)

    @classmethod
    def glob_fs_path(cls, path: Union[str, Path]) -> List[Path]:
        "Expands a path like /my/**/*/file.png into all matching files."

        if cls.is_archive_path(path):
            raise ValueError(f"path cannot be a path to data inside archive files: {path}")
        # FIXME: windows?
        rootless_glob = Path("/".join(Path(path).absolute().parts[1:])).as_posix()
        return cls.sort_paths(list(Path("/").glob(rootless_glob)))

    @classmethod
    def glob(
        cls, path: Union[Path, str], separator: str = os.path.pathsep + NOT_FOLLOWED_BY_DOUBLE_SLASH
    ) -> List[Path]:
        urls = []
        for p in re.split(separator, Path(path).as_posix()):
            path_item: Path = Path(p)
            if cls.is_remote(path_item):
                urls.append(path_item)
            elif cls.is_archive_path(path_item):
                urls += cls.glob_archive_path(path_item)
            else:
                urls += cls.glob_fs_path(path_item)
        return urls  # do not resort so colon-separate globs maintain provided order
