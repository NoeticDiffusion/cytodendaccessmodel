"""Download the candidate shallow-plane file for sub-656228 / ses-1245548523.

Tries asset IDs in order from lowest acquisition ID (likely shallowest plane)
upward. Downloads to the canonical path. Probes the ROI count after download.

Run once and let it complete in the background.
"""
import sys, json, urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
DEST_DIR = ROOT / "data" / "dandi" / "raw" / "000871" / "sub-656228"
DEST_DIR.mkdir(parents=True, exist_ok=True)

# Ordered candidates: try acq-1245937725 first (lowest ID = likely shallowest)
CANDIDATES = [
    ("1245937725", "d1cac429-7064-4141-8326-e59866db4cc3"),
    ("1245937727", "c7fd6217-cc9a-4427-9543-c36e95c1478d"),
    ("1245937728", "e3dc86b9-143e-4e52-bbbd-9ebee333151b"),
    ("1245937730", "7382bf5e-f2b4-460a-a265-b216ea5e64af"),
    ("1245937731", "09ee7436-114d-4141-9353-82a58a77f8ef"),
    ("1245937733", "8674c7ee-5dcb-43de-a3f2-baf3cb1f103f"),
    ("1245937734", "5d9224c7-59d6-4b2a-a877-9e572fca33a9"),
]

def get_s3_url(asset_id: str) -> str:
    url = f"https://api.dandiarchive.org/api/assets/{asset_id}/download/"
    req = urllib.request.Request(url, headers={"Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.url

def download_file(s3_url: str, dest: Path, chunk_mb: int = 16) -> None:
    req = urllib.request.Request(s3_url)
    downloaded = 0
    with urllib.request.urlopen(req, timeout=60) as r:
        total = int(r.headers.get("Content-Length", 0))
        with open(dest, "wb") as f:
            while True:
                chunk = r.read(chunk_mb * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                pct = downloaded / total * 100 if total else 0
                print(f"\r  {downloaded/1024/1024:.0f} / {total/1024/1024:.0f} MB  ({pct:.1f}%)", end="", flush=True)
    print()

def probe_rois(path: Path) -> int:
    """Return ROI count; raises if file has no usable dff traces."""
    import h5py
    with h5py.File(str(path), "r") as hf:
        ophys = hf.get("processing/ophys")
        if ophys is None:
            raise ValueError("no processing/ophys found")
        if "dff" not in ophys:
            avail = list(ophys.keys())
            raise ValueError(f"'dff' not found in ophys. Available: {avail}")
        dff = ophys["dff"]
        # Find the traces dataset
        traces_key = next(
            (k for k in dff.keys() if "traces" in k.lower() or "roi_response" in k.lower()),
            None,
        )
        if traces_key is None:
            raise ValueError(f"no traces dataset found in dff. Keys: {list(dff.keys())}")
        shape = dff[traces_key].shape
        return shape[1] if len(shape) >= 2 else 0

def probe_depth(path: Path) -> str:
    import h5py
    with h5py.File(str(path), "r") as hf:
        try:
            desc = hf["general/optophysiology/imaging_plane_1/description"][()]
            return desc.decode() if isinstance(desc, bytes) else str(desc)
        except Exception:
            return "unknown"

for acq_id, asset_id in CANDIDATES:
    fname = f"sub-656228_ses-1245548523-acq-{acq_id}_image+ophys.nwb"
    dest = DEST_DIR / fname
    if dest.exists():
        print(f"{fname} already exists ({dest.stat().st_size // (1024*1024)} MB)")
        n = probe_rois(dest)
        d = probe_depth(dest)
        print(f"  ROIs: {n}  depth: {d}")
        break
    print(f"Downloading {fname} ...")
    try:
        s3_url = get_s3_url(asset_id)
        download_file(s3_url, dest)
        n = probe_rois(dest)
        d = probe_depth(dest)
        print(f"Download complete. ROIs: {n}  depth: {d}")
        break
    except Exception as e:
        print(f"  Failed: {e}")
        if dest.exists():
            dest.unlink()
        continue
