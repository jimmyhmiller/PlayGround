const { promisify }  = require("util");
const exec = promisify(require("child_process").exec);
const gitDiffParser = require('git-diff-parser');
const { flatMap, some } = require("lodash");
const { file: tempFile } = require("tmp-promise");
const {readFile, writeFile, mkdir} = require('fs/promises');
const homedir = require('os').homedir();
const pathLib = require('path');


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

  const result = await exec(`git diff ${oldPath} ${path}`)
    .catch(x => x.stdout);

  const cleanedPatch = result
    .replace(new RegExp(path, 'g'), oldPath)
    .replace(new RegExp(root, 'g'), "");

  // This structure is a bit much, but maybe makes sense?
  // I will have to try things out a bit.
  // I will definitely have to figure out the right patch granularity.
  // What if some comments can apply and others can't?
  const pathMinusHome = root.replace(`${homedir}/`, "");
  console.log(`${homedir}/.personal_notes/${pathMinusHome}/${pathLib.dirname(fileName)}`);
  await mkdir(`${homedir}/.personal_notes/${pathMinusHome}/${pathLib.dirname(fileName)}`, {recursive: true})
  await writeFile(`${homedir}/.personal_notes/${pathMinusHome}/${fileName}.patch`, cleanedPatch);


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
  // 1 just do directory (useful for patches)
  // 2 `git rev-list --parents HEAD | tail -1`


  // Need to think about granularity when it comes to storing patches
  // Also have to think about patch order.
  // Can a patch not apply because I applied a different note patch?
  // Maybe it is a patch for filename and you can't make new patches
  // if you didn't first turn on personal? Or something like that?

}

main();