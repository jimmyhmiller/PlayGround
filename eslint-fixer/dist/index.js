'use strict';

var _extends = Object.assign || function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; };

require('babel-polyfill');

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

var _globPromise = require('glob-promise');

var _globPromise2 = _interopRequireDefault(_globPromise);

var _findConfig = require('find-config');

var _findConfig2 = _interopRequireDefault(_findConfig);

var _ignore = require('ignore');

var _ignore2 = _interopRequireDefault(_ignore);

var _ora = require('ora');

var _ora2 = _interopRequireDefault(_ora);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

var _marked = [getErrors].map(regeneratorRuntime.mark);

var spinner = (0, _ora2.default)('Checking for eslint violations');

var prompt = (0, _promptSync2.default)({ sigint: true });

var helpText = '\nEslint Fixer - a fast way to fix eslint errors\n\nUsage:\n  eslint-fix [--args=<ESLINT_ARGS>] [--exts=<FILE_EXTENSIONS>] <path>\n  eslint-fix -h | --help\n  eslint-fix --version\n\nOptions:\n  <path>        Path to run eslint (default=\'.\')\n  -h --help     Show this screen.\n  --version     Show version.\n  --args        Pass args to eslint\n  --exts        Comma separated string of file extensions ".js,.jsx"';

var clearConsole = function clearConsole() {
  console.log('\x1Bc');
};

var newLine = function newLine() {
  console.log('');
};

var editFile = function editFile(_ref) {
  var file = _ref.file,
      line = _ref.line,
      column = _ref.column;

  var editor = process.env.EDITOR || 'nano';
  (0, _child_process.execSync)(editor + ' ' + file + ':' + line + ':' + column); // figure out how to make it work with all editors
};

var hashProblem = function hashProblem(_ref2) {
  var file = _ref2.file,
      line = _ref2.line,
      column = _ref2.column;
  return file + ':' + line + ':' + column;
};

var getProblemInfo = function getProblemInfo(problem, skipped) {
  var file = problem.filePath;
  var source = problem.source;
  var error = (0, _compat.first)((0, _compat.dropWhile)(problem.messages, function (m) {
    return skipped.has(hashProblem(_extends({ file: file }, m)));
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
  try {
    return (0, _child_process.execSync)(command);
  } catch (e) {
    return e.stdout;
  }
};

var getProblems = function getProblems(_ref3) {
  var file = _ref3.file,
      _ref3$args = _ref3.args,
      args = _ref3$args === undefined ? "" : _ref3$args;

  return JSON.parse(execIgnoreExitCode(process.cwd() + '/node_modules/eslint/bin/eslint.js --format=json ' + args + ' ' + file).toString('utf-8'));
};

var getFiles = function getFiles(path, args) {
  return getProblems({ file: path, args: args }).filter(function (p) {
    return p.errorCount > 0 || p.warningCount > 0;
  }).map(function (p) {
    return p.filePath;
  });
};

var getNextProblem = function getNextProblem(_ref4) {
  var file = _ref4.file,
      args = _ref4.args,
      skippedProblems = _ref4.skippedProblems;

  var problem = (0, _compat.first)(getProblems({ args: args, file: file }));
  return getProblemInfo(problem, skippedProblems);
};

var fixProblem = function fixProblem(_ref5) {
  var file = _ref5.file,
      line = _ref5.line,
      column = _ref5.column;

  editFile({ file: file, line: line, column: column });
};

var showSpinner = function showSpinner() {
  clearConsole();
  spinner.start();
};

var stopSpinner = function stopSpinner() {
  spinner.stop();
};

var showProblem = function showProblem(_ref6) {
  var file = _ref6.file,
      line = _ref6.line,
      column = _ref6.column,
      source = _ref6.source,
      message = _ref6.message;

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

function getErrors(_ref7) {
  var file = _ref7.file,
      args = _ref7.args;
  var skippedProblems, problem, skipped;
  return regeneratorRuntime.wrap(function getErrors$(_context) {
    while (1) {
      switch (_context.prev = _context.next) {
        case 0:
          skippedProblems = new Set();

        case 1:
          if (!true) {
            _context.next = 11;
            break;
          }

          problem = getNextProblem({ file: file, args: args, skippedProblems: skippedProblems });

          if (problem) {
            _context.next = 5;
            break;
          }

          return _context.abrupt('break', 11);

        case 5:
          _context.next = 7;
          return problem;

        case 7:
          skipped = _context.sent;

          if (skipped) {
            skippedProblems.add(hashProblem(problem));
          }
          _context.next = 1;
          break;

        case 11:
        case 'end':
          return _context.stop();
      }
    }
  }, _marked[0], this);
}

var processErrors = function processErrors(args, file) {
  var errors = getErrors({ file: file, args: args['--args'] });
  var _iteratorNormalCompletion = true;
  var _didIteratorError = false;
  var _iteratorError = undefined;

  try {
    for (var _iterator = errors[Symbol.iterator](), _step; !(_iteratorNormalCompletion = (_step = _iterator.next()).done); _iteratorNormalCompletion = true) {
      var error = _step.value;

      showProblem(error);
      var action = getAction();
      processAction(action, error, errors);
    }
  } catch (err) {
    _didIteratorError = true;
    _iteratorError = err;
  } finally {
    try {
      if (!_iteratorNormalCompletion && _iterator.return) {
        _iterator.return();
      }
    } finally {
      if (_didIteratorError) {
        throw _iteratorError;
      }
    }
  }
};

var main = function main() {

  var DEFAULT_IGNORE_DIRS = ["node_modules", "bower_components"];

  var args = _neodoc2.default.run(helpText, { smartOptions: true });
  var path = args['<path>'] || '.';
  var exts = (args['--exts'] || "").replace(',', '|') || '.js';

  // const eslintIgnore = (findConfig.read('.eslintignore') || "").split('\n');
  // const removeIgnored = files => ignore().add(DEFAULT_IGNORE_DIRS).add(eslintIgnore).filter(files);


  showSpinner();

  var filesPromise = Promise.resolve(getFiles(path, args['--args']));
  filesPromise.then(function (files) {
    return files.forEach(processErrors.bind(null, args));
  }).catch(function (e) {
    return console.error(e);
  }).then(stopSpinner);
};

main();