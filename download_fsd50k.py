# Download the FSD50K sound effect dataset
# https://zenodo.org/records/5117901
# Usage: python3 download_fsd50k.py

from helpers import DownloadProgressBar
import os
from urllib.request import urlretrieve
import zipfile


def download(url, filename):
    """Download the file at URL, and save it as 'datasets/filename'"""
    print(f"Downloading {url}, this may take a while...")
    urlretrieve(url, f"datasets/{filename}", DownloadProgressBar())


def extract(filename):
    """Extract the zip file at 'datasets/filename'"""
    with zipfile.ZipFile(f"datasets/{filename}", "r") as zipref:
        zipref.extractall("datasets")


def remove(filenames):
    """Delete a list of files"""
    for filename in filenames:
        os.remove(f"datasets/{filename}")


def download_multipart_zip(urls, intermediary_file_name):
    """Download a multipart zip file from the given urls, and give the user advice on how to unzip it"""
    # Filenames of each part - named .z01, .z02, etc., with the final ending in .zip
    parts_filenames = [
        f"{intermediary_file_name}_multipart.z0{i+1}" for i in range(len(urls) - 1)
    ]
    parts_filenames.append(f"{intermediary_file_name}_multipart.zip")

    # Download all files
    for i, url in enumerate(urls):
        download(url, parts_filenames[i])

    # Give user advice on extracting multipart zip file
    print("======")
    print(f"Downloaded zip files to {', '.join(parts_filenames)}.")
    print(
        "Python does not support extracting multi-part zip files, so you'll have to do this manually. Try running:"
    )
    print(
        f"\tzip -s 0 datasets/{parts_filenames[-1]} --out datasets/{intermediary_file_name}_all.zip && unzip datasets/{intermediary_file_name}_all.zip -d datasets"
    )


def main():
    """
    Download FSD50K dataset
    """
    # URL for CSV files with metadata
    ground_truth_url = "https://zenodo.org/records/5117901/files/ARCA23K-FSD.ground_truth.zip?download=1"

    # URL for actual audio files
    # These are multipart ZIP files, so must be combined into a single zip file
    # before extracting!
    audio_dev_files_urls = [
        "https://zenodo.org/records/4060432/files/FSD50K.dev_audio.z01?download=1",
        "https://zenodo.org/records/4060432/files/FSD50K.dev_audio.z02?download=1",
        "https://zenodo.org/records/4060432/files/FSD50K.dev_audio.z03?download=1",
        "https://zenodo.org/records/4060432/files/FSD50K.dev_audio.z04?download=1",
        "https://zenodo.org/records/4060432/files/FSD50K.dev_audio.z05?download=1",
        "https://zenodo.org/records/4060432/files/FSD50K.dev_audio.zip?download=1",
    ]

    audio_eval_files_urls = [
        "https://zenodo.org/records/4060432/files/FSD50K.eval_audio.z01?download=1",
        "https://zenodo.org/records/4060432/files/FSD50K.eval_audio.zip?download=1",
    ]

    # Create the datasets directory
    os.makedirs("datasets", exist_ok=True)

    # Download train/validation sets
    download_multipart_zip(audio_dev_files_urls, "audio_dev")

    # Download test set
    download_multipart_zip(audio_eval_files_urls, "audio_eval")


if __name__ == "__main__":
    # Fix for macOS, see https://docs.python.org/3/library/urllib.request.html
    os.environ["no_proxy"] = "*"

    # Download files!
    main()
