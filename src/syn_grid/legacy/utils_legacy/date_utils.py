import datetime


def get_date() -> str:
    """Return current timestamp in consistent format."""
    return datetime.datetime.now().astimezone().strftime("%y%m%d_%H-%M-%S")
