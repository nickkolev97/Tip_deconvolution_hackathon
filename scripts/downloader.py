import requests
import os
import zipfile

def download_file(url, output_filename):
    """
    Downloads a file from the given URL and saves it to the specified output filename.

    :param url: The URL to download the file from.
    :param output_filename: The local filename to save the downloaded file.
    """
    try:
        print(f"Starting download from {url}...")
        
        # Send a GET request to the URL
        response = requests.get(url, stream=True)
        
        # Check if the request was successful
        response.raise_for_status()

        # Save the file locally in chunks
        with open(output_filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive chunks
                    file.write(chunk)

        print(f"Download complete. File saved as {output_filename}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

def unzip_file(zip_path, extract_to):
    """
    Unzips the specified zip file to the given directory.

    :param zip_path: Path to the zip file.
    :param extract_to: Directory to extract the contents to.
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            print(f"Extracted {zip_path} to {extract_to}")
    except zipfile.BadZipFile:
        print(f"Error: {zip_path} is not a valid zip file.")

if __name__ == "__main__":
    # URL of the file to download
    url = "https://zenodo.org/api/records/7664070/files-archive"

    extract_folder = os.path.join("data", "raw_data")
    output_filename = os.path.join(extract_folder, "files-archive.zip")

    # Check if raw_data STM folder does not exist or is empty
    stm_extract_folder = os.path.join(extract_folder, "STM")
    if not os.path.exists(stm_extract_folder) or not os.listdir(stm_extract_folder):
        print(f"Folder {stm_extract_folder} does not exist or is empty. Proceeding with download and extraction...")
        # Ensure the extraction folder exists
        os.makedirs(extract_folder, exist_ok=True)
        
        if not os.path.exists(output_filename):
            download_file(url, output_filename)
        else:
            print(f"File {output_filename} already exists. Skipping download.")

        # Unzip the main archive
        unzip_file(output_filename, extract_folder)

        # Check for STM.zip inside the extracted folder and unzip it
        stm_zip_path = os.path.join(extract_folder, "STM.zip")
        if os.path.exists(stm_zip_path):
            os.makedirs(stm_extract_folder, exist_ok=True)
            unzip_file(stm_zip_path, stm_extract_folder)
            
            # Delete STM.zip after extraction
            os.remove(stm_zip_path)
            print(f"Deleted {stm_zip_path}")
        else:
            print(f"STM.zip not found in {extract_folder}.")

        # Delete files-archive.zip after extraction
        if os.path.exists(output_filename):
            os.remove(output_filename)
            print(f"Deleted {output_filename}")
    else:
        print(f"Folder {stm_extract_folder} already exists and is not empty. Skipping download and extraction.")
