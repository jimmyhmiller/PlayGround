#!/usr/bin/env node

var stdin = process.openStdin();

var data = "";

stdin.on('data', function(chunk) {
  data += chunk;
});

stdin.on('end', function() {
  console.log(eval(data));
});
