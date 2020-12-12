from google_drive_downloader import GoogleDriveDownloader as gdd

class Adapter():
    @staticmethod
    def download(dest_path:str, file_id, *args, **kwargs):
        gdd.download_file_from_google_drive(file_id, dest_path, **kwargs)