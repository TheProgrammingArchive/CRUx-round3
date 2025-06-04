import zipfile
with zipfile.ZipFile('celebrity-face-image-dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('')