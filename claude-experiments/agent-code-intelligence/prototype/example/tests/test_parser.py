from src.app.config import Config
from src.app.parser import Parser
from src.app.server import handle_request


def test_precedence():
    assert handle_request("1 + 2 * 3", Config()) == 7


def test_parens():
    assert handle_request("( 1 + 2 ) * 3", Config()) == 9


def test_min_bp_cutoff():
    parser = Parser("1 + 2")
    assert parser.parse_expression(100) == 1


def test_config_timeout_default():
    assert Config().timeout == 30
