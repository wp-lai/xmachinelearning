import os
from urllib.request import urlretrieve


def maybe_download(filename, expected_bytes, url, data_dir=None):
    """Download a file if not present, and make sure it's the right size."""
    if not data_dir:
        filename = os.path.join(data_dir, filename)
    if not os.path.exists(filename):
        filename, _ = urlretrieve(url + filename, filename)

    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print(f"Found and verified {filename}")
    else:
        print(statinfo.st_size)
        raise Exception(f"Failed to verify {filename}.")
    return filename
