const { promisify }  = require("util");
const exec = promisify(require("child_process").exec);
const parseDiff = require('parse-diff');

const main = async () => {
  // @personal
  // Should this be staged or just diff?
  // Why not both?
  const response = await exec("git diff");
  console.log(JSON.stringify(parseDiff(response.stdout), null, 4))
}

main();