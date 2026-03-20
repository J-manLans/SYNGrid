import os

def get_asset_path(*relative_path_parts: str) -> str:
    """
    Returns an absolute path to a resource inside the package,
    works whether installed editable or normally.
    """

    base_dir = os.path.dirname(os.path.abspath(__file__))
    package_root = os.path.abspath(os.path.join(base_dir, ".."))
    return os.path.join(package_root, *relative_path_parts)