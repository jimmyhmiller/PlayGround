const { send, json } = require('micro');
const { router, get, post } = require('microrouter');
const streamToPromise = require('stream-to-promise');
const crypto = require('crypto');
const MemoryStream = require('memorystream');
const UrlPattern = require('url-pattern');
const path = require('path');
require('dotenv').config()

var fs = require('fs');
var manta = require('manta');

var client = manta.createClient({
    sign: manta.privateKeySigner({
        key: process.env.MANTA_KEY,
        keyId: process.env.MANTA_KEY_ID,
        user: process.env.MANTA_USER
    }),
    user: process.env.MANTA_USER,
    url: process.env.MANTA_URL
});
console.log('manta ready: %s', client.toString());


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

const putObject = (filePath, data) => {
  const json = JSON.stringify(data);
  var opts = {
      md5: crypto.createHash('md5').update(json).digest('base64'),
      size: Buffer.byteLength(json),
      type: 'application/json'
  };
  var stream = new MemoryStream();
  return createDirectory(filePath)
    .then(_ => {
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

const getDatesStatus = async (req, res) => {
  try {
    console.log("here")
    const { team, user, date } = req.params;
    const data = await getObject(`standup/${team}/${date}/${user}.json`)
    send(res, 200, data)
  } catch (e) {
    handleKeyNotFound(e, send.bind(null, res))
  }
}

const postDatesStatus = async (req, res) => {
  try {
    const body = await json(req)
    const { team, user, date } = req.params;
    await putObject(`standup/${team}/${date}/${user}.json`, body)
    send(res, 201)
  } catch (e) {
    console.log(e)
    send(res, 500, e)
  }
}

const notfound = (req, res) => send(res, 404, {message: 'Not found'})

// default of router includes dots, which is dangerous
const urlPatternOpts = {
  segmentValueCharset: "a-zA-Z0-9-"
}

module.exports = router(
  get(new UrlPattern('/standup/:team/:date/:user', urlPatternOpts), getDatesStatus),
  post(new UrlPattern('/standup/:team/:date/:user', urlPatternOpts), postDatesStatus),
  get("/*", notfound),
)