#!/usr/bin/python

import argparse
import code
import string
import sys

from pyDatalog import pyDatalog, pyEngine, pyParser, util
from datalog import Datalog, kvp

#################################################
##   Command Line Tool for Interactive Queries ##
#################################################

def import_all_from(module_name):
    mod = __import__(module_name)
    for member_name in dir(mod):
        globals()[member_name] = getattr(mod, member_name)

def globalize_atoms(code):
    for name in code.co_names:
        if name in globals():
            if not isinstance(globals()[name], (pyParser.Symbol, pyParser.Variable)):
                raise util.DatalogError("Name conflict. Can't redefine %s as atom" %
                                        name, None, None)
        else:
            if name[0] not in string.ascii_uppercase:
                globals()[name] = pyParser.Symbol(name)
            else:
                globals()[name] = pyParser.Variable(name)

def exec_datalog(source):
    code = compile(source, '<string>', 'single')
    with pyParser.ProgramContext():
        newglobals = {}
        pyParser.add_symbols(code.co_names, newglobals)
        globalize_atoms(code)
        [exec(code) in newglobals]

class fdbqueryConsole(code.InteractiveConsole):
    valid_modes = ['query', 'python']

    def set_mode(self,mode):
        assert mode in fdbqueryConsole.valid_modes
        self.mode = mode
        sys.ps1 = mode+'> '

    def interpolate(self, source):
        # ugly string interpolation
        return """
exec_datalog('''
%s
''')
""" % source

    def runsource(self, source, filename='console', symbol='single'):

        if source in fdbqueryConsole.valid_modes:
            self.set_mode(source)
            return

        if self.mode == 'query':
            new_source = self.interpolate(source)
        elif source.lstrip().startswith('qry:'):
            source = source.lstrip().lstrip('qry:').lstrip()
            new_source = self.interpolate(source)
        else:
            new_source = source

        try:
            code.InteractiveConsole.runsource(self, new_source, filename, symbol)
        except Exception as e:
            print(e)

pyEngine.Auto_print = True

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='FoundationDB Query Console')
    parser.add_argument('-p', '--python', help='''Python module to be imported.
        pyDatalog.create_atoms must be called for any Datalog included.''')
    parser.add_argument('-d', '--datalog', help='''File with Datalog statements
        (only) to be loaded. Atoms will be automatically created.''')
    args = parser.parse_args()
    if args.python:
        import_all_from(args.python)
    if args.datalog:
        with open(args.datalog, 'r') as f:
            dl_defs = f.read()
        f.closed
        pyDatalog.load(dl_defs)
        globalize_atoms(compile(dl_defs, '<string>', 'exec'))

    console = fdbqueryConsole(locals=locals())
    console.set_mode('query')
    console.interact('')
