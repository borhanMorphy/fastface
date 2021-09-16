import logging
import os
import tarfile
import zipfile

from tqdm import tqdm

# add logging

logger = logging.getLogger("fastface.adapter")


class ExtractHandler:
    @staticmethod
    def extract(
        file_path: str, dest_path: str, *args, remove_after: bool = True, **kwargs
    ):
        logger.info("extracting {} to {}".format(file_path, dest_path))
        if file_path.endswith(".zip"):
            ExtractHandler._extract_zipfile(file_path, dest_path, *args, **kwargs)
        else:
            # expected .tar file
            ExtractHandler._extract_tarfile(file_path, dest_path, *args, **kwargs)

        # clear the file
        if remove_after:
            logger.warning("removing source {} file".format(file_path))
            os.remove(file_path)

    @staticmethod
    def _extract_tarfile(file_path: str, dest_path: str, set_attrs: bool = False):
        if file_path.endswith(".tar.gz") or file_path.endswith(".tgz"):
            mode = "r:gz"
        elif file_path.endswith(".tar.bz2") or file_path.endswith(".tbz"):
            mode = "r:bz2"
        else:
            raise AssertionError("tar file extension is not valid")

        with tarfile.open(file_path, mode=mode) as foo:
            members = foo.getmembers()
            for member in tqdm(
                members, desc="extracting tar.gz file to {}".format(dest_path)
            ):
                try:
                    foo.extract(member, path=dest_path, set_attrs=set_attrs)
                except PermissionError:
                    pass  # ignore
                except Exception as e:
                    logger.error(
                        "extracing member: {} failed with\n{}".format(member, e)
                    )

    @staticmethod
    def _extract_zipfile(file_path: str, dest_path: str):
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(dest_path)
