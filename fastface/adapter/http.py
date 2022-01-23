import os

import requests
from tqdm import tqdm

from .extract_handler import ExtractHandler


class HttpAdapter:
    @staticmethod
    def download(
        dest_path: str,
        file_name: str = None,
        url: str = None,
        extract: bool = False,
        **kwargs
    ):
        # TODO check if file name format is matched with downloaded file

        assert isinstance(url, str), "url must be string but found:{}".format(type(url))

        file_name = url.split("/")[-1] if file_name is None else file_name
        res = requests.get(url, stream=True)

        assert (
            res.status_code == 200
        ), "wrong status code \
            recived:{} with response:{}".format(
            res.status_code, res.content
        )

        total = int(res.headers.get("content-length", 0))

        file_path = os.path.join(dest_path, file_name)

        if not os.path.exists(dest_path):
            os.makedirs(dest_path, exist_ok=True)

        with open(file_path, "wb") as foo, tqdm(
            desc=file_path,
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in res.iter_content(chunk_size=1024):
                size = foo.write(data)
                bar.update(size)

        if not extract:
            return

        ExtractHandler.extract(file_path, dest_path, **kwargs)
