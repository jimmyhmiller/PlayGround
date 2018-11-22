const streamToPromise = require('stream-to-promise');
const crypto = require('crypto');
const MemoryStream = require('memorystream');
const path = require('path');
require('dotenv').config()

const fs = require('fs');
const manta = require('@jimmyhmiller/manta');

const createClient = () => manta.createClient({
  sign: manta.privateKeySigner({
      key: process.env.MANTA_KEY,
      keyId: process.env.MANTA_KEY_ID,
      user: process.env.MANTA_USER
  }),
  user: process.env.MANTA_USER,
  url: process.env.MANTA_URL
});


const createDirectory = (filePath) => {
  return new Promise((resolve, reject) => {
    client.mkdirp(`~~/stor/${path.dirname(filePath)}`, function (err) {
      if (err) {
        reject(err);
      } else {
        resolve();
      }
    });
  })
}

const objectExists = (filePath) => {
  return new Promise((resolve, reject) => {
    client.info(`~~/stor/${filePath}`, function (err, data) {
      if (err) {
        reject(err);
      } else {
        resolve(data)
      }
    });
  })
}

const getObject = (filePath) => {
  return new Promise((resolve, reject) => {
    client.get(`~~/stor/${filePath}`, function (err, stream) {
      if (err) {
        reject(err);
      } else {
        stream.setEncoding('utf8');
        streamToPromise(stream)
          .then(data => resolve(JSON.parse(data.toString())))
      }
    });
  })
}

const listDirectory = (filePath) => {
  return new Promise((resolve, reject) => {
    const stuff = [];
    client.ls(`~~/stor/${filePath}`, function (err, res) {
      if (err) {
        reject(err);
      } else {
       res.on('object', function (obj) {
         stuff.push(obj);
       });

       res.on('directory', function (dir) {
         stuff.push(dir);
       });

       res.once('end', function () {
         resolve(stuff);
       });
     }
    });
  })
}

const putObject = (filePath, data) => {
  const json = JSON.stringify(data);
  var opts = {
      md5: crypto.createHash('md5').update(json).digest('base64'),
      size: Buffer.byteLength(json),
      type: 'application/json',
      copies: 1
  };

  // determine when to createDirectory
  var stream = new MemoryStream();
  return new Promise((resolve, reject) => {
    client.put(`~~/stor/${filePath}`, stream, opts, function (err) {
      if (err) {
        reject(err)
      } else {
        resolve()
      }
    });
    stream.end(json);
  })
}

const handleKeyNotFound = (e, cb) => {
  if (e.code === "ResourceNotFound") {
    cb(404, {error: 'Status not found'})
  } else {
    console.log(e)
    cb(500, {error: 'Unexpected error'})
  }
}



module.exports = {
  createDirectory,
  objectExists,
  getObject,
  listDirectory,
  putObject,
  handleKeyNotFound,
  createClient,
}
