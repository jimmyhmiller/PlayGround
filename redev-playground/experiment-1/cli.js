#!/usr/bin/env node
const fs = require('fs');
const readline = require('readline');
const mimeTypes = require('mime-types');
const fetch = require("node-fetch");
const { readFile } = require('fs').promises;
const path = require('path');
const yargs = require('yargs/yargs')
const { hideBin } = require('yargs/helpers')

const argv = yargs(hideBin(process.argv)).argv;
const fileName = argv._[0];
const contentType = mimeTypes.lookup(fileName);




const bodyByContentType = ({ contentType, fileContents, url, onBuild }) => {
  if (contentType === "application/javascript") {
    return {
      url,
      onBuild,
      execute: fileContents.toString('utf8'),
    }
  } else {
    return {
      url,
      onBuild,
      body: fileContents.toString('utf8'),
      headers: {
        "Content-Type": contentType
      }
    }
  }
}

const urlForFileName = ({ fileName, contentType }) => {
  let url = fileName;
  url = url.startsWith("index") ? "" : url;
  url = contentType === "application/javascript" ? `api/${path.parse(fileName).name}` : url;
  url = `/${url}`
  return url
}

const processCommand = async () => {
  if (contentType) {
    const onBuild = argv.onbuild ? (await readFile(argv.onbuild)).toString('utf-8') : undefined;
    const fileContents = await readFile(fileName);
    const url = urlForFileName({ contentType, fileName })
    const request = bodyByContentType({ contentType, fileContents, url, onBuild });
    const resp = await fetch(`http://localhost:3000/routes`, {
      method: "POST",
      body: JSON.stringify(request)
    })
    console.log(`http://localhost:3000${request.url}`)
  }
}

processCommand()
