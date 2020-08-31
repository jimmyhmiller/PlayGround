const { promisify }  = require("util");
const exec = promisify(require("child_process").exec);
const gitDiffParser = require('git-diff-parser');
const { flatMap, some } = require("lodash");
const { file: tempFile } = require("tmp-promise");
const readFile = promisify(require('fs').readFile);
const writeFile = promisify(require('fs').writeFile);


const getGitRoot = async () => {
  const root = await exec("git rev-parse --show-toplevel");

  return root.stdout.trim();
}

const getFilePatch = async (fileName) => {
  const {fd, path, cleanup} = await tempFile();
  const root = await getGitRoot();
  const oldPath = `${root}/${fileName}`;
  const fileContents = (await readFile(oldPath)).toString('utf8');
  await writeFile(path, fileContents);
  const strippedFile = fileContents
    // Think about windows new lines?
    .split("\n")
    .reduce(({personal, lines, prefix}, line) => {
      if (personal) {
        const stayPersonal = line.trim().substring(0, prefix.length) === prefix;
        return {
          personal: stayPersonal,
          lines: stayPersonal ? lines : lines.concat([line]),
          prefix: stayPersonal ? prefix : undefined,
        }
      } else if (line.includes("@personal")) {
        return {
          personal: true,
          lines,
          prefix: line.substring(0, line.indexOf("@personal")).trim()
        }
      } else {
        return {personal: false, lines: lines.concat([line])}
      }
    }, {personal: false, lines: []})
    .lines
    .join("\n");

  await writeFile(oldPath, strippedFile);
    const result = await exec(`git diff ${oldPath} ${path}`)
      .catch(x => x.stdout);
  // Need to write this to some home directory location
  await writeFile(`${root}/test.patch`, result);

  await writeFile(oldPath, strippedFile);
  // console.log("not actually changing files");
  // await writeFile(oldPath, fileContents);

  // console.log(strippedFile);
  cleanup();
}

const main = async () => {
  // @personal
  // Should this be staged or just diff?
  // Why not both?
  const response = gitDiffParser((await exec("git diff")).stdout);
  // console.log(JSON.stringify(response));

  let files = flatMap(response.commits, 'files')
    // Adding this so I don't ruin my own source code.
    .filter(f => !f.name.startsWith("personal-notes-app"))
    .filter(f => some(f.lines, line => line.type === "added" && line.text.includes("@personal")))
    .map(f => f.name)
    .map(getFilePatch)





  // Edit patch to be about same file at root of git
  // Save patch
  // Now we can check later if the patch would apply cleanly.
  // If so, the note is still valid for current commit,
  // if not, the note is on old commit
  // Need to save date and time.
  
  // to identify repo I could either
  // 1 just do directory (kind of stinks)
  // 2 `git rev-list --parents HEAD | tail -1`

}

main();