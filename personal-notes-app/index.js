const { promisify }  = require("util");
const exec = promisify(require("child_process").exec);
const parseDiff = require('parse-diff');
const gitDiffParser = require('git-diff-parser');
const { flatMap, some } = require("lodash");

const main = async () => {
  // @personal
  // Should this be staged or just diff?
  // Why not both?
  const response = gitDiffParser((await exec("git diff")).stdout);

  let files = flatMap(response.commits, 'files')
    // Adding this so I don't ruin my own source code.
    .filter(f => !f.name.startsWith("personal-notes-app"))
    .filter(f => some(f.lines, line => line.text.includes("@personal")))



  console.log(JSON.stringify(files.map(f => f.name), null, 4))


  // Copy files to tmp
  // Make copy and remove @personal notes
  // Record patch and save it.
  // Now we can check later if the patch would apply cleanly.
  // If so, the note is still valid for current commit,
  // if not, the note is on old commit

}

main();