import "babel-polyfill";
import { execSync, spawnSync } from 'child_process';
import neodoc from 'neodoc';
import { first, dropWhile } from 'zaphod/compat';
import codeFrame from 'babel-code-frame';
import chalk from 'chalk';
import promptSync from 'prompt-sync';
import glob from 'glob-promise';
import findConfig from 'find-config';
import ignore from 'ignore';
import ora from 'ora';

const spinner = ora('Checking for eslint violations');

const prompt = promptSync({ sigint: true });

const helpText = `
Eslint Fixer - a fast way to fix eslint errors

Usage:
  eslint-fix <path>
  eslint-fix -h | --help
  eslint-fix --version
  eslint-fix --args[=ESLINT_ARGS]
  eslint-fix --exts[=FILE_EXTENSIONS]

Options:
  <path>        Path to run eslint (default='.')
  -h --help     Show this screen.
  --version     Show version.
  --args        Pass args to eslint
  --exts        Comma separated string of file extensions ".js,.jsx"`

const clearConsole = () => {
  console.log('\x1Bc');
}

const newLine = () => {
  console.log('');
}

const editFile = ({ file, line, column }) => {
  const editor = process.env.EDITOR || 'nano';
  execSync(`${editor} ${file}:${line}:${column}`) // figure out how to make it work with all editors
}

const hashProblem = ({ file, line, column }) => `${file}:${line}:${column}`

const getProblemInfo = (problem, skipped) => {
  const file = problem.filePath;
  const source = problem.source;
  const error = first(dropWhile(problem.messages, m => skipped.has(hashProblem({file, ...m}))));
  if (!error) {
    return undefined;
  }
  const line = error.line;
  const column = error.column;
  const message = error.message;
  return { file, line, column, source, message };
}

const getNextProblem = ({ file, args="", skippedProblems }) => {
  const problem = first(JSON.parse(execSync(`./node_modules/eslint/bin/eslint.js --format=json ${args} ${file}`).toString('utf-8')));
  return getProblemInfo(problem, skippedProblems);
}

const fixProblem = ({ file, line, column }) => {
  editFile({ file, line, column });
}

const showSpinner = () => {
  clearConsole();
  spinner.start();
}

const stopSpinner = () => {
  spinner.stop();
}

const showProblem = ({ file, line, column, source, message }) => {
  stopSpinner();
  const result = codeFrame(source, line, column, { highlightCode: true });
  clearConsole();
  console.log(chalk.white(`${file}:${line}:${column}`))
  newLine();
  console.log(result);
  newLine();
  console.log(message);
}

const getAction = () => {
  return prompt('(e = edit [default], s = skip, a = abort)? ');
}

const skipLast = errors => errors.next(true)

const processAction = (action, problemInfo, errors) => {
  if (action === 'e') {
    fixProblem(problemInfo)
  } else if (action === 's') {
    skipLast(errors);
  } else if (action === 'a') {
    process.exit();
  } else {
    fixProblem(problemInfo)
  }
  showSpinner();
}

function* getErrors({ file, args }) {
  let skippedProblems = new Set();
  while (true) {
    const problem = getNextProblem({file, args, skippedProblems});
    if (!problem) {
      break;
    }
    const skipped = yield problem;
    if (skipped) {
      skippedProblems.add(hashProblem(problem))
    }
  }
}

const processErrors = (args, file) => {
  const errors = getErrors({ file, args: args['--args'] })
  for (const error of errors) {
    showProblem(error);
    const action = getAction();
    processAction(action, error, errors);
  }
}

const main = () => {

  const DEFAULT_IGNORE_DIRS = [
    "node_modules",
    "bower_components"
  ];

  const args = neodoc.run(helpText)
  const path = args['<path>'] || '.';
  const exts = (args['--exts'] || "").replace(',', '|') || '.js';

  const eslintIgnore = findConfig.read('.eslintignore').split('\n');
  const removeIgnored = files => ignore().add(DEFAULT_IGNORE_DIRS).add(eslintIgnore).filter(files);
  const filesPromise = glob(`${path}/**/*@(${exts})`)

  showSpinner();
  filesPromise
    .then(removeIgnored)
    .then(files => files.forEach(processErrors.bind(null, args)))
    .catch(e => console.error(e))
    .then(stopSpinner)
}

main();
