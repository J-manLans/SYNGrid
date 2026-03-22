import os


def get_package_path(*relative_path_parts: str) -> str:
    """
    Returns an absolute path to a resource inside the package,
    works whether installed editable or normally.
    """
    # current file's directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # package root (one level up)
    package_root = os.path.abspath(os.path.join(base_dir, ".."))
    # build absolute path from package root
    return os.path.join(package_root, *relative_path_parts)


def get_package_root() -> str:
    """
    Returns an absolute path to the package root,
    works whether installed editable or normally.
    """
    # current file's directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # package root (one level up)
    package_root = os.path.abspath(os.path.join(base_dir, ".."))
    # return root
    return os.path.join(package_root)


def get_project_path(*relative_path_parts: str) -> str:
    """
    Returns an absolute path relative to the current working directory.
    """

    base_dir = os.getcwd()
    return os.path.join(base_dir, *relative_path_parts)
