#!/usr/bin/env python3
"""
Download ModelNet40 dataset for PointNet training/evaluation.

Based on: https://github.com/yanx27/Pointnet_Pointnet2_pytorch
"""

import urllib.request
import zipfile
import argparse
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download a file with progress bar."""
    print(f"Downloading: {url}")
    print(f"To: {output_path}")
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_modelnet40(output_dir):
    """Download and extract ModelNet40 dataset."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ModelNet40 aligned version (commonly used for PointNet)
    url = "https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip"
    zip_path = output_dir / "modelnet40_normal_resampled.zip"
    extract_dir = output_dir / "modelnet40_normal_resampled"
    
    # Check if already downloaded
    if extract_dir.exists():
        print(f"✓ ModelNet40 already exists at: {extract_dir}")
        return
    
    # Download
    try:
        download_url(url, zip_path)
        print("✓ Download complete")
    except Exception as e:
        print(f"✗ Download failed: {e}")
        print("\nAlternative download methods:")
        print("1. Manual download from: https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip")
        print(f"2. Save to: {zip_path}")
        print("3. Run this script again to extract")
        return
    
    # Extract
    print("\nExtracting...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"✓ Extracted to: {extract_dir}")
        
        # Clean up zip file
        zip_path.unlink()
        print("✓ Cleaned up zip file")
        
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return
    
    print("\n✓ ModelNet40 dataset ready!")
    print(f"  Location: {extract_dir}")
    print(f"  Classes: 40")
    print(f"  Train samples: ~9,840")
    print(f"  Test samples: ~2,468")


def download_shapenet_part(output_dir):
    """Download ShapeNet Part Segmentation dataset."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    url = "https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip"
    zip_path = output_dir / "shapenet_part.zip"
    extract_dir = output_dir / "shapenetcore_partanno_segmentation_benchmark_v0_normal"
    
    if extract_dir.exists():
        print(f"✓ ShapeNet Part already exists at: {extract_dir}")
        return
    
    print("\nShapeNet Part Segmentation Dataset:")
    print(f"  URL: {url}")
    print(f"  Size: ~346 MB")
    print("\nNote: This dataset is used for part segmentation tasks.")
    print("For this demo, we'll use synthetic data instead.")


def main():
    parser = argparse.ArgumentParser(description='Download PointNet datasets')
    parser.add_argument('--dataset', type=str, default='modelnet40',
                        choices=['modelnet40', 'shapenet'],
                        help='Dataset to download')
    parser.add_argument('--output_dir', type=str, default='assets/datasets',
                        help='Output directory for datasets')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PointNet Dataset Downloader")
    print("=" * 60)
    print()
    
    if args.dataset == 'modelnet40':
        download_modelnet40(args.output_dir)
    elif args.dataset == 'shapenet':
        download_shapenet_part(args.output_dir)


if __name__ == '__main__':
    main()

