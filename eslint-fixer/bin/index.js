#!/usr/bin/env node
'use strict';

var _set = require('babel-runtime/core-js/set');

var _set2 = _interopRequireDefault(_set);

var _asyncGenerator2 = require('babel-runtime/helpers/asyncGenerator');

var _asyncGenerator3 = _interopRequireDefault(_asyncGenerator2);

var _regenerator = require('babel-runtime/regenerator');

var _regenerator2 = _interopRequireDefault(_regenerator);

var _asyncToGenerator2 = require('babel-runtime/helpers/asyncToGenerator');

var _asyncToGenerator3 = _interopRequireDefault(_asyncToGenerator2);

var _promise = require('babel-runtime/core-js/promise');

var _promise2 = _interopRequireDefault(_promise);

var _extends2 = require('babel-runtime/helpers/extends');

var _extends3 = _interopRequireDefault(_extends2);

var getFiles = function () {
  var _ref5 = (0, _asyncToGenerator3.default)(_regenerator2.default.mark(function _callee(path, args) {
    var problems;
    return _regenerator2.default.wrap(function _callee$(_context) {
      while (1) {
        switch (_context.prev = _context.next) {
          case 0:
            _context.next = 2;
            return getProblems({ file: path, args: args });

          case 2:
            problems = _context.sent;
            return _context.abrupt('return', problems.filter(function (p) {
              return p.errorCount > 0 || p.warningCount > 0;
            }).map(function (p) {
              return p.filePath;
            }));

          case 4:
          case 'end':
            return _context.stop();
        }
      }
    }, _callee, this);
  }));

  return function getFiles(_x, _x2) {
    return _ref5.apply(this, arguments);
  };
}();

var getNextProblem = function () {
  var _ref6 = (0, _asyncToGenerator3.default)(_regenerator2.default.mark(function _callee2(_ref7) {
    var file = _ref7.file,
        args = _ref7.args,
        skippedProblems = _ref7.skippedProblems;
    var problem;
    return _regenerator2.default.wrap(function _callee2$(_context2) {
      while (1) {
        switch (_context2.prev = _context2.next) {
          case 0:
            _context2.t0 = _compat.first;
            _context2.next = 3;
            return getProblems({ args: args, file: file });

          case 3:
            _context2.t1 = _context2.sent;
            problem = (0, _context2.t0)(_context2.t1);
            return _context2.abrupt('return', getProblemInfo(problem, skippedProblems));

          case 6:
          case 'end':
            return _context2.stop();
        }
      }
    }, _callee2, this);
  }));

  return function getNextProblem(_x3) {
    return _ref6.apply(this, arguments);
  };
}();

var getErrors = function () {
  var _ref10 = _asyncGenerator3.default.wrap(_regenerator2.default.mark(function _callee3(_ref11) {
    var file = _ref11.file,
        args = _ref11.args;
    var skippedProblems, problem, skipped;
    return _regenerator2.default.wrap(function _callee3$(_context3) {
      while (1) {
        switch (_context3.prev = _context3.next) {
          case 0:
            skippedProblems = new _set2.default();

          case 1:
            if (!true) {
              _context3.next = 13;
              break;
            }

            _context3.next = 4;
            return _asyncGenerator3.default.await(getNextProblem({ file: file, args: args, skippedProblems: skippedProblems }));

          case 4:
            problem = _context3.sent;

            if (problem) {
              _context3.next = 7;
              break;
            }

            return _context3.abrupt('break', 13);

          case 7:
            _context3.next = 9;
            return problem;

          case 9:
            skipped = _context3.sent;

            if (skipped) {
              skippedProblems.add(hashProblem(problem));
            }
            _context3.next = 1;
            break;

          case 13:
          case 'end':
            return _context3.stop();
        }
      }
    }, _callee3, this);
  }));

  return function getErrors(_x4) {
    return _ref10.apply(this, arguments);
  };
}();

var processErrors = function () {
  var _ref12 = (0, _asyncToGenerator3.default)(_regenerator2.default.mark(function _callee4(args, file) {
    var errors;
    return _regenerator2.default.wrap(function _callee4$(_context4) {
      while (1) {
        switch (_context4.prev = _context4.next) {
          case 0:
            _context4.next = 2;
            return getErrors({ file: file, args: args['--args'] });

          case 2:
            errors = _context4.sent;
            return _context4.abrupt('return', forEach(errors, function (error) {
              showProblem(error);
              var action = getAction();
              processAction(action, error, errors);
            }));

          case 4:
          case 'end':
            return _context4.stop();
        }
      }
    }, _callee4, this);
  }));

  return function processErrors(_x5, _x6) {
    return _ref12.apply(this, arguments);
  };
}();

var _child_process = require('child_process');

var _neodoc = require('neodoc');

var _neodoc2 = _interopRequireDefault(_neodoc);

var _compat = require('zaphod/compat');

var _babelCodeFrame = require('babel-code-frame');

var _babelCodeFrame2 = _interopRequireDefault(_babelCodeFrame);

var _chalk = require('chalk');

var _chalk2 = _interopRequireDefault(_chalk);

var _promptSync = require('prompt-sync');

var _promptSync2 = _interopRequireDefault(_promptSync);

var _ora = require('ora');

var _ora2 = _interopRequireDefault(_ora);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

var spinner = (0, _ora2.default)('Checking for eslint violations');

var prompt = (0, _promptSync2.default)({ sigint: true });

var helpText = '\nEslint Fixer - a fast way to fix eslint errors\n\nUsage:\n  eslint-fixer [--args=<ESLINT_ARGS>] <path>\n  eslint-fixer -h | --help\n  eslint-fixer --version\n\nOptions:\n  <path>        Path to run eslint\n  -h --help     Show this screen.\n  --version     Show version.\n  --args        Pass args to eslint';

var clearConsole = function clearConsole() {
  console.log('\x1Bc');
};

var newLine = function newLine() {
  console.log('');
};

var filloutEditorTemplate = function filloutEditorTemplate(str, _ref) {
  var file = _ref.file,
      line = _ref.line,
      column = _ref.column;

  var regex = function regex(name) {
    return new RegExp('%' + name);
  };
  return str.replace(regex("file"), file).replace(regex("line"), line).replace(regex("column"), column);
};

var editFile = function editFile(_ref2) {
  var file = _ref2.file,
      line = _ref2.line,
      column = _ref2.column;

  var editorTemplate = process.env.ESLINT_FIXER_EDITOR || 'nano +%line,%column %file';
  var command = filloutEditorTemplate(editorTemplate, { file: file, line: line, column: column }).split(" ");
  var editor = command[0];
  var args = command.slice(1);
  (0, _child_process.spawnSync)(editor, args, {
    stdio: 'inherit',
    shell: true
  });
};

var hashProblem = function hashProblem(_ref3) {
  var file = _ref3.file,
      line = _ref3.line,
      column = _ref3.column;
  return file + ':' + line + ':' + column;
};

var getProblemInfo = function getProblemInfo(problem, skipped) {
  var file = problem.filePath;
  var source = problem.source;
  var error = (0, _compat.first)((0, _compat.dropWhile)(problem.messages, function (m) {
    return skipped.has(hashProblem((0, _extends3.default)({ file: file }, m)));
  }));
  if (!error) {
    return undefined;
  }
  var line = error.line;
  var column = error.column;
  var message = error.message;
  return { file: file, line: line, column: column, source: source, message: message };
};

var execIgnoreExitCode = function execIgnoreExitCode(command) {
  return new _promise2.default(function (resolve) {
    (0, _child_process.exec)(command, { maxBuffer: 2000 * 1024 }, function (e, stdout, stderr) {
      return resolve(stdout.toString());
    });
  });
};

var tryParseJsonOrAbort = function tryParseJsonOrAbort(str) {
  try {
    return JSON.parse(str);
  } catch (e) {
    console.log(str);
    process.exit();
  }
};

var getProblems = function getProblems(_ref4) {
  var file = _ref4.file,
      _ref4$args = _ref4.args,
      args = _ref4$args === undefined ? "" : _ref4$args;

  return execIgnoreExitCode('eslint --format=json ' + args + ' ' + file).then(tryParseJsonOrAbort);
};

var fixProblem = function fixProblem(_ref8) {
  var file = _ref8.file,
      line = _ref8.line,
      column = _ref8.column;

  editFile({ file: file, line: line, column: column });
};

var showSpinner = function showSpinner() {
  clearConsole();
  spinner.start();
};

var stopSpinner = function stopSpinner() {
  spinner.stop();
};

var showProblem = function showProblem(_ref9) {
  var file = _ref9.file,
      line = _ref9.line,
      column = _ref9.column,
      source = _ref9.source,
      message = _ref9.message;

  stopSpinner();
  var result = (0, _babelCodeFrame2.default)(source, line, column, { highlightCode: true });
  clearConsole();
  console.log(_chalk2.default.white(file + ':' + line + ':' + column));
  newLine();
  console.log(result);
  newLine();
  console.log(message);
};

var getAction = function getAction() {
  return prompt('(e = edit [default], s = skip, a = abort)? ');
};

var skipLast = function skipLast(errors) {
  return errors.next(true);
};

var processAction = function processAction(action, problemInfo, errors) {
  if (action === 'e') {
    fixProblem(problemInfo);
  } else if (action === 's') {
    skipLast(errors);
  } else if (action === 'a') {
    process.exit();
  } else {
    fixProblem(problemInfo);
  }
  showSpinner();
};

function forEach(ai, fn) {
  return ai.next().then(function (r) {
    if (!r.done) {
      fn(r.value);
      return forEach(ai, fn);
    }
  });
}

var main = function main() {

  var args = _neodoc2.default.run(helpText, { smartOptions: true });
  var path = args['<path>'];

  if (!path) {
    console.log(helpText.trim());
    process.exit();
  }

  showSpinner();

  var filesPromise = getFiles(path, args['--args']);

  filesPromise.then(function (files) {
    return _promise2.default.all(files.map(processErrors.bind(null, args)));
  }).catch(function (e) {
    return console.error(e);
  }).then(stopSpinner).then(function () {
    return console.log('No errors found.');
  });
};

main();