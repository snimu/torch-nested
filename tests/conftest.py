import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """This allows us to check for these params in sys.argv."""
    parser.addoption("--overwrite", action="store_true", default=False)
