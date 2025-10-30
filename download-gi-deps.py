import os
import urllib.request
import zipfile
import shutil

# URLs for missing dependencies
dll_sources = {
    "z.dll": "https://github.com/winlibs/zlib/releases/download/zlib-1.3/zlib-1.3-msvc-x64.zip",
    "intl-8.dll": "https://github.com/mlocati/gettext-iconv-windows/releases/download/v0.21-v1.16/gettext0.21-iconv1.16-static-64.zip"
}

download_dir = "gtk_deps_download"
os.makedirs(download_dir, exist_ok=True)

# Download and extract dependencies
for dll_name, url in dll_sources.items():
    zip_path = os.path.join(download_dir, f"{dll_name}.zip")
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, zip_path)
    
    print(f"Extracting {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(download_dir)

# Copy DLLs to GTK bin directory
gtk_bin = r'C:\gtk\bin'
for root, dirs, files in os.walk(download_dir):
    for file in files:
        if file.lower().endswith('.dll'):
            source = os.path.join(root, file)
            dest = os.path.join(gtk_bin, file)
            print(f"Copying {source} to {dest}")
            shutil.copy2(source, dest)
            
            # Also create lib* version
            lib_dest = os.path.join(gtk_bin, f"lib{file}")
            print(f"Creating lib version at {lib_dest}")
            shutil.copy2(source, lib_dest)

print("Done installing dependencies!")
