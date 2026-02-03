// This script monitors dashboard events and writes params to file
const fs = require('fs');
const path = require('path');

const paramsFile = path.join(__dirname, 'params.json');

// Default params
let currentParams = {
  gravity: 0.6,
  jumpStrength: -10,
  pipeSpeed: 3,
  pipeGap: 150,
  pipeInterval: 90,
  birdSize: 20,
  birdX: 100
};

// Write params to file
function writeParams(params) {
  fs.writeFileSync(paramsFile, JSON.stringify(params, null, 2));
  console.log('Wrote params:', params);
}

// Listen on stdin for param updates (JSON)
process.stdin.on('data', (data) => {
  try {
    const params = JSON.parse(data.toString());
    currentParams = params;
    writeParams(params);
  } catch (e) {
    console.error('Error parsing params:', e.message);
  }
});

console.log('Params sync ready. Send JSON params via stdin.');
writeParams(currentParams);
