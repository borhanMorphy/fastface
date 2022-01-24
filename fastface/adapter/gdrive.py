import os

import gdown

from .extract_handler import ExtractHandler


class GoogleDriveAdapter:
    @staticmethod
    def download(
        dest_path: str,
        file_name: str = None,
        file_id: str = None,
        extract: bool = False,
        sub_dir: str = "",
        **kwargs
    ):
        assert isinstance(file_id, str), "file id must be string but found:{}".format(
            type(file_id)
        )
        assert isinstance(
            file_name, str
        ), "file name must be string but found:{}".format(type(file_name))

        dest_path = os.path.join(dest_path, sub_dir)

        if not os.path.exists(dest_path):
            os.makedirs(dest_path, exist_ok=True)

        file_path = os.path.join(dest_path, file_name)

        gdown.download(
            output=file_path,
            quiet=False,
            use_cookies=False,
            id=file_id,
        )

        if not extract:
            return

        ExtractHandler.extract(file_path, dest_path, **kwargs)
