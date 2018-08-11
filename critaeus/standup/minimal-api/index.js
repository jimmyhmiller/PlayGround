const { send, json } = require('micro')
const { router, get, post } = require('microrouter')
const s3 = new (require('aws-sdk/clients/s3'))();

const bucket = 'a-bucket-that-i-dont-care-about';

const writeToS3 = (bucket, key, data) => {
  return s3.putObject({
    Bucket: bucket,
    Key: key,
    Body: JSON.stringify(data),
  }).promise()
}

const readFromS3 = (bucket, key) => {
  return s3.getObject({
    Bucket: bucket,
    Key: key,
  })
  .promise()
  .then(data => JSON.parse(data.Body.toString()))
}

const handleKeyNotFound = (e, cb) => {
  if (e.code === "NoSuchKey") {
    cb(400, {error: 'Status not found'})
  } else {
    console.log(e)
    cb(500, {error: 'Unexpected error'})
  }
}

const getDatesStatus = async (req, res) => {
  try {
    const { team, user, date } = req.params;
    const data = await readFromS3(bucket, `${team}/${date}/${user}.json`)
    send(res, 200, data)
  } catch (e) {
    handleKeyNotFound(e, send.bind(null, res))
  }
}

const postDatesStatus = async (req, res) => {
  try {
    const body = await json(req)
    const { team, user, date } = req.params;
    await writeToS3(bucket, `${team}/${date}/${user}.json`, body)
    send(res, 201)
  } catch (e) {
    console.log(e)
    send(res, 500, e)
  }
}


module.exports = router(
  get('/standup/:team/:date/:user', getDatesStatus),
  post('/standup/:team/:date/:user', postDatesStatus),
)