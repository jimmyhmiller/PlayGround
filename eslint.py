import sys
import json
import os

from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import TerminalFormatter
import re

def clear_terminal():
    print(chr(27) + "[2J")

def replace_zeros(m):
    return " " * len(m.group())

def each_line(f, string):
    return "\n".join(f(line) for line in string.split("\n"))

def get_source(error):
    lexer = get_lexer_by_name("javascript", stripall=True)
    formatter = TerminalFormatter(linenos=True)
    formatter._lineno = error["line"] - 1
    result = highlight(error["source"], lexer, formatter)
    result = each_line(lambda s: re.sub("^0+", replace_zeros, s), result)
    return result

def has_warning_or_error(lint):
    return lint["errorCount"] > 0 or lint["warningCount"] > 0

flatten = lambda l: [item for sublist in l for item in sublist]

def set_file_path(error, message):
    message["filePath"] = error["filePath"]
    return message

def get_inner_error(error):
    return [set_file_path(error, message) for message in error["messages"]]

def get_lint_error(error):
    return "   " + error["message"]

def get_file_info(error):
    return "{}:{}:{}".format(error["filePath"], error["line"], error["column"])

def get_actions(error):
    return "\n(e = edit, i=ignore [default])"

def process_actions(action, error):
    if action == "e":
        editor = os.environ.get('EDITOR') or 'nano'
        os.system("{} {}:{}:{}".format(editor, error["filePath"], error["line"], error["column"]))
    elif action == "i":
        pass

# filename = sys.argv[1]
lint_info = json.load(sys.stdin)
sys.stdin = open("/dev/tty") # without this raw_input would fail

errors = filter(has_warning_or_error, lint_info)
errors = flatten(get_inner_error(error) for error in errors)


for error in errors:
    clear_terminal()
    print(get_file_info(error))
    print(get_source(error))
    print(get_lint_error(error))
    print(get_actions(error))
    print("\n" * 4)
    action = raw_input()
    process_actions(action, error)

