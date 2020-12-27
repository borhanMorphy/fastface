import requests
import zipfile
import os

class HttpAdapter():
    @staticmethod
    def download(dest_path:str, file_name:str=None, url:str=None, unzip:bool=False, **kwargs):
        # TODO check if file name format is matched with downloaded file
        assert isinstance(url, str), f"url must be string but found:{type(url)}"
        file_name = url.split("/")[-1] if file_name is None else file_name
        res = requests.get(url)
        assert res.status_code == 200,f"wrong status code recived:{res.status_code} with response:{res.content}"
        file_path = os.path.join(dest_path,file_name)
        if not os.path.exists(dest_path):
            os.makedirs(dest_path, exist_ok=True)
        with open(file_path, "wb") as foo:
            foo.write(res.content)

        if not unzip: return

        with zipfile.ZipFile(file_path,"r") as zip_ref:
            zip_ref.extractall(dest_path)