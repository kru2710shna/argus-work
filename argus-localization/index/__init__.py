"""FlatIndex is FaissFlatIndex when faiss is installed, else the numpy
equivalent (same exact search, different on-disk format)."""

try:
    from index.faiss_index import FaissFlatIndex as FlatIndex
except ImportError:
    from index.numpy_index import NumpyFlatIndex as FlatIndex

__all__ = ["FlatIndex"]
