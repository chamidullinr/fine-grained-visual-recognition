import json
import os
import shutil
import zipfile
import tarfile


def to_json(dict_: dict, filename: str):
    """Save dictionary to JSON file."""
    with open(filename, 'w') as f:
        json.dump(dict_, f)


def read_json(filename: str) -> dict:
    """Load dictionary from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def is_filetype(filename: str, filetype: str) -> bool:
    """Check if the filename ends with the given file extension like ".json" or ".csv"."""
    return filename.endswith(filetype)


def get_filenames_in_dir(path: str, filetype: str = None, include_path: bool = True) -> list:
    """
    Return list of files contained in the given directory.

    Args:
        path: directory path
        filetype: file extension to filter, when specified keep only files with the given filetype
        include_path: include path as prefix in returned filenames
    """
    try:
        files = [os.path.join(path, item) if include_path else item
                 for item in os.listdir(path)]
    except FileNotFoundError:
        files = []

    if filetype is not None:
        files = [x for x in files if is_filetype(x, filetype)]
    return files


def create_path(path: str):
    """Create directories in the given path if it does not exist."""
    dir_path = os.path.dirname(path)
    if len(dir_path) > 0:
        os.makedirs(dir_path, exist_ok=True)
    # from pathlib import Path
    # Path(dir_path).mkdir(parents=True, exist_ok=True)


def copy_file(src: str, trg: str):
    """Copy file from source directory to target directory."""
    create_path(trg)
    shutil.copyfile(src, trg)


def create_archive(filename: str, files: list):
    """
    Create archive file (zip or tar.gz).

    Args:
        filename: archive file to create
        files: list of files to archive
    """
    if not isinstance(files, (list, tuple)):
        files = [files]
    if is_filetype(filename, '.zip'):
        with zipfile.ZipFile(filename, 'w') as zip_:
            for file in files:
                zip_.write(file)
    elif is_filetype(filename, '.tar.gz'):
        with tarfile.open(filename, 'w:gz') as tar:
            for file in files:
                tar.add(file)
    else:
        raise ValueError(
            f'Unknown file extension (type) in {filename}. '
            'Allowed file extensions (types) are ".zip" or ".tar.gz"')


def extract_archive(filename: str, trg_path: str):
    """
    Extract files from the archive file (zip or tar.gz).

    Args:
        filename: archive file to extract from
        trg_path: destination path of extracted files
    """
    create_path(trg_path)
    if is_filetype(filename, '.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_:
            zip_.extractall(trg_path)
    elif is_filetype(filename, '.tar.gz'):
        with tarfile.open(filename) as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, trg_path)
    else:
        raise ValueError(
            f'Unknown file extension (type) in {filename}. '
            'Allowed file extensions (types) are ".zip" or ".tar.gz"')
