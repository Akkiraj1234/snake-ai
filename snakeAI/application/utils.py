from pathlib import Path


BASE_ASSETS_DIR = Path(__file__).parent.parent.parent

class Theme:
    def __init__(self, values: dict[str, tuple[int, int, int]]) -> None:
        self._values = values

    def __getattr__(self, name: str):
        return self._values[name]