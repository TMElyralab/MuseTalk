import os
import sys
import importlib
import subprocess

# List any packages needed for downloading
REQUIRED_PACKAGES = [
    "huggingface_hub",
    "gdown",
    "requests"
]

PYTHON_EXEC = sys.executable

# Install missing packages BEFORE importing
for pkg in REQUIRED_PACKAGES:
    try:
        importlib.import_module(pkg.replace("-", "_"))
    except ImportError:
        print(f"Installing {pkg} ...")
        subprocess.run([PYTHON_EXEC, "-m", "pip", "install", pkg])


from huggingface_hub import hf_hub_download
import sys
import subprocess
import importlib

# --- Configuration ---
CHECKPOINTS_DIR = "models"
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co") # Use mirror if set

# --- Directory Setup ---
DIRS = [
     "musetalkV15", "syncnet", "dwpose",
    "face-parse-bisent", "sd-vae", "whisper", "musetalk" # Ensure 'musetalk' is here if V1.0 is needed
]

for d in DIRS:
    os.makedirs(os.path.join(CHECKPOINTS_DIR, d), exist_ok=True)
print(f"✅ Created base directory: {CHECKPOINTS_DIR} and subdirectories.")

# --- Hugging Face Downloads ---

def download_hf_files(repo_id, filenames, subdir="", has_subpath=False):
    """
    Downloads a list of files from a Hugging Face repo.

    If has_subpath is True (e.g., MuseTalk), files are downloaded relative to CHECKPOINTS_DIR.
    If has_subpath is False (e.g., Whisper), files are downloaded directly into CHECKPOINTS_DIR/subdir.
    """
    target_local_dir = os.path.join(CHECKPOINTS_DIR, subdir)

    # If the filename contains the directory structure (e.g., "repo_name/file.bin"),
    # we need to set local_dir to CHECKPOINTS_DIR to preserve the path.
    # Otherwise, we set local_dir to the final destination (target_local_dir).
    final_local_dir = CHECKPOINTS_DIR if has_subpath else target_local_dir

    for filename in filenames:
        print(f"Downloading {filename} from {repo_id} to {target_local_dir}...")

        # Use hf_hub_download. The output path handling is based on `has_subpath`.
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=final_local_dir,
            endpoint=HF_ENDPOINT
        )
    print(f"✅ Finished downloading files for {repo_id} into {subdir}/.")


# 1. MuseTalk V1.0 & V1.5 Weights (Uses subpaths in filenames)
# NOTE: The repo files are structured like "musetalk/..." and "musetalkV15/..."
# Setting local_dir=CHECKPOINTS_DIR ensures this internal structure is preserved under "models/"

# V1.0 Files (Target: models/musetalk)
download_hf_files(
    repo_id="TMElyralab/MuseTalk",
    filenames=[
        "musetalk/musetalk.json",
        "musetalk/pytorch_model.bin"
    ],
    subdir="musetalk",
    has_subpath=True # Filenames contain the subdir path
)
# V1.5 Files (Target: models/musetalkV15)
download_hf_files(
    repo_id="TMElyralab/MuseTalk",
    filenames=[
        "musetalkV15/musetalk.json",
        "musetalkV15/unet.pth"
    ],
    subdir="musetalkV15",
    has_subpath=True # Filenames contain the subdir path
)

# 2. SD VAE Weights (No subpaths in filenames)
# Target: models/sd-vae/
download_hf_files(
    repo_id="stabilityai/sd-vae-ft-mse",
    filenames=[
        "config.json",
        "diffusion_pytorch_model.bin",
        "diffusion_pytorch_model.safetensors"
    ],
    subdir="sd-vae",
    has_subpath=False
)

# 3. Whisper Weights (No subpaths in filenames)
# FIX: This now downloads directly into models/whisper/
# Target: models/whisper/
download_hf_files(
    repo_id="openai/whisper-tiny",
    filenames=[
        "config.json",
        "pytorch_model.bin",
        "preprocessor_config.json"
    ],
    subdir="whisper",
    has_subpath=False
)

# 4. DWPose Weights (No subpaths in filenames)
# Target: models/dwpose/
download_hf_files(
    repo_id="yzd-v/DWPose",
    filenames=["dw-ll_ucoco_384.pth"],
    subdir="dwpose",
    has_subpath=False
)

# 5. SyncNet Weights (No subpaths in filenames)
# Target: models/syncnet/
download_hf_files(
    repo_id="ByteDance/LatentSync",
    filenames=["latentsync_syncnet.pt"],
    subdir="syncnet",
    has_subpath=False
)

print("--- Hugging Face downloads complete. ---")



# Download BiSeNet Face Parse Model file (from Google Drive)
try:
    import gdown
except ImportError:
    subprocess.run(['pip', 'install', 'gdown'])
    import gdown
gdown.download(
    'https://drive.google.com/uc?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812',
    os.path.join(CHECKPOINTS_DIR, "face-parse-bisent", "79999_iter.pth"),
    quiet=False
)




# Download resnet18 model
import requests
url = "https://download.pytorch.org/models/resnet18-5c106cde.pth"
output_path = os.path.join(CHECKPOINTS_DIR, "face-parse-bisent", "resnet18-5c106cde.pth")
response = requests.get(url, stream=True)
with open(output_path, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)
print(f"✅ Downloaded {url} to {output_path}")



s3fd_url = "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"
s3fd_dest_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
os.makedirs(s3fd_dest_dir, exist_ok=True)
s3fd_dest_path = os.path.join(s3fd_dest_dir, "s3fd-619a316812.pth")

if not os.path.exists(s3fd_dest_path) or os.path.getsize(s3fd_dest_path) < 85_000_000:  # ~85MB expected
    print(f"Downloading S3FD weights from {s3fd_url} ...")
    response = requests.get(s3fd_url, stream=True)
    with open(s3fd_dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"✅ Downloaded S3FD model to {s3fd_dest_path}")
else:
    print(f"✅ S3FD weights already present: {s3fd_dest_path}")
    
print("--- All model downloads complete. ---")