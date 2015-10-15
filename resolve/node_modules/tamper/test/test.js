var libpath = process.env.TAMPER_COV ? '../lib-cov' : '../lib'
  , tamper = require(libpath + '/tamper')
  , fs = require('fs')
  , http = require('http')
  , parse = require('url').parse
  , qs = require('querystring')
  , request = require('supertest')
  , assert = require('assert')

// Test-content to be replaced by the middleware
var content = 'Content to be replaced'

// Replaces all occurrences of the test-content
function replace(body, req, headers, finished) {
  updatedBody = body.replace(new RegExp(content, 'g'), 'Replaced content');
  if(finished) {
	finished(updatedBody)
  } else {
    return updatedBody
  }
}

// A versatile handler to test the various ServerResponse methods
function handler(req, res) {
  var url = parse(req.url)
    , q = qs.parse(url.query)
    , headers = JSON.parse(q.headers || null)

  // call writeHead() if `status` or `headers` parameter is present
  if (q.status || headers) {
    res.writeHead(q.status || 200, headers)
  }

  // pipe the request to the response in case of POST requests
  if (req.method == 'POST') {
    req.pipe(res)
  }
  else {
    if (q.fill) {
      // write a buffer filled with 'x'
      var b = new Buffer(parseInt(q.fill, 10))
      b.fill('x')
      res.write(b)
    }

    // call write() with the content of the `write` parameter
    if (q.write) res.write(q.write)

    // call end() with the content of the `end` parameter
    res.end(q.end)
  }
}

// Creates a HttpServer with the given middleware
function createServer(middleware) {
  return http.createServer(function(req, res) {
    middleware(req, res, function(err) {
      handler(req, res)
    })
  })
}

// Returns a function that can be passed to describe().
function test(app, expected) {
  return function() {
    it('should echo the body upon POST', function(done) {
      request(app)
        .post('/')
        .send(content)
        .expect(200)
        .expect(expected, done)
    })
    it('should echo the write parameter upon GET', function(done) {
      request(app)
        .get('/')
        .query({write: content})
        .expect(200)
        .expect(expected, done)
    })
    it('should echo the end parameter upon GET', function(done) {
      request(app)
        .get('/')
        .query({end: content})
        .expect(200)
        .expect(expected, done)
    })
    it('should work with large responses', function(done) {
      request(app)
        .get('/')
        .query({fill: 65536, end: content})
        .expect(200)
        .end(function(err, res) {
          assert.equal(res.text.length, 65536 + expected.length)
          done()
        })
    })
    it('should echo both parameters', function(done) {
      request(app)
        .get('/')
        .query({write: content, end: content})
        .expect(200)
        .expect(expected+expected, done)
    })
    it('should set the status', function(done) {
      request(app)
        .get('/')
        .query({status: 202})
        .expect(202, done)
    })
    it('should set headers', function(done) {
      request(app)
        .get('/')
        .query({headers: '{"X-Works": "Yes"}'})
        .expect('X-Works', 'Yes')
        .expect(200, done)
    })
    it('should honor the X-Tamper header', function(done) {
      request(app)
        .get('/')
        .query({write: content, headers: '{"X-Tamper": "No"}'})
        .expect(content, done)
    })
  }
}

// A server without the tamper middleware
var plain = createServer(function(req, res, next) { next(req, res) })

// A server with a tamper middleware that does nothing
var inactive = createServer(tamper(function() {}))

// A server that replaces content unless the X-Tamper header is set to 'No'
var active = createServer(tamper(function(req, res) {
  return res.getHeader('X-Tamper') == 'No' ? false : replace
}))

describe('The test handler', test(plain, content))
describe('A present but inactive middleware', test(inactive, content))
describe('A tampering middleware', test(active, replace(content)))
