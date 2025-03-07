# Download the ARCA sound effect dataset
# https://zenodo.org/records/5117901
# Usage: python3 download_arca.py

import os
import zipfile
from urllib.request import urlretrieve


def download(url, filename):
    """
    Download the file at URL, and save it as 'datasets/filename'
    """
    print(f"Downloading {url}, this may take a while...")
    urlretrieve(url, f"datasets/{filename}")


def extract(filename):
    """
    Extract the zip file at 'datasets/filename'
    """
    with zipfile.ZipFile(f"datasets/{filename}", "r") as zipref:
        zipref.extractall("datasets")


def concatenate(in_filenames, out_filename):
    """
    Merge a list of files into one file
    """
    with open(f"datasets/{out_filename}", "wb") as out_file:
        # Write each in_file to the out_file
        for in_filename in in_filenames:
            with open(f"datasets/{in_filename}", "rb") as in_file:
                out_file.write(in_file.read())


def remove(filenames):
    """
    Delete a list of files
    """
    for filename in filenames:
        os.remove(f"datasets/{filename}")


def main():
    """
    Download ARCA dataset
    """
    # URL for CSV files with metadata
    ground_truth_url = "https://zenodo.org/records/5117901/files/ARCA23K-FSD.ground_truth.zip?download=1"

    # URL for actual audio files
    # These are multipart ZIP files, so must be combined into a single zip file
    # before extracting!
    audio_files_urls = [
        "https://zenodo.org/records/5117901/files/ARCA23K.audio.z01?download=1",
        "https://zenodo.org/records/5117901/files/ARCA23K.audio.z02?download=1",
        "https://zenodo.org/records/5117901/files/ARCA23K.audio.z03?download=1",
        "https://zenodo.org/records/5117901/files/ARCA23K.audio.z04?download=1",
        "https://zenodo.org/records/5117901/files/ARCA23K.audio.zip?download=1"
    ]

    # Create the datasets directory
    os.makedirs("datasets", exist_ok=True)

    # Download CSV metadata
    download(ground_truth_url, "ground_truth.zip")
    extract("ground_truth.zip")
    remove(["ground_truth.zip"])

    # Download audio files
    for i, url in enumerate(audio_files_urls):
        download(url, f"audio_data_{str(i)}.z0{i+1}")

    # Combine multipart zip files into a single zip file
    audio_zip_filenames = [f"audio_data_{str(i)}.z0{i+1}" for i in range(len(audio_files_urls))]
    concatenate(audio_zip_filenames, "audio_data_all.zip")

    # Extract the audio data
    # NOTE : old versions of Python don't support multiple disks in a zip file, which is why this may fail
    try:
        extract("audio_data_all.zip")
        remove(["audio_data_all.zip"])
    except:
        print("Downloaded audio data to datasets/audio_data_all.zip, you'll have to manually extract it! Try running 'unzip datasets/audio_data_all.zip -d datasets'")

    # Remove intermediary files
    remove(audio_zip_filenames)


if __name__ == "__main__":
    # Fix for macOS, see https://docs.python.org/3/library/urllib.request.html
    os.environ["no_proxy"] = "*"

    # Download files!
    main()
