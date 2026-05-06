from pathlib import Path


def get_package_path(*relative_path_parts: str) -> Path:
    """
    Returns an absolute path to a file inside the package,
    works whether installed editable or normally.
    """

    return _get_package_root() / Path(*relative_path_parts)


def get_project_path(*relative_path_parts: str) -> Path:
    """
    Returns an absolute path relative to the current working directory.
    """

    return Path.cwd() / Path(*relative_path_parts)


# ================= #
#      Helpers      #
# ================= #


def _get_package_root() -> Path:
    """
    Returns an absolute path to the package root,
    works whether installed editable or normally.
    """

    return Path(__file__).parent.parent
