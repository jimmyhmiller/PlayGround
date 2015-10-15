# tamper
## Node.js middleware to capture and modify response bodies

[![Build Status](https://secure.travis-ci.org/fgnass/tamper.png)](http://travis-ci.org/fgnass/tamper)

The intention of this module is to allow development tools like
[Yeoman](http://yeoman.io) to inject code snippets into the files they serve.

## Installation

  npm install tamper

## Usage

```javascript
var tamper = require('tamper');

app.use(tamper(function(req, res) {

  // Look at the request or the response headers and decide what to do.

  // In this case we only want to modify html responses:
  if (res.getHeader('Content-Type') != 'text/html') {

    // When returning a falsy value processing will continue as usual
    // without any performance impact.
    return
  }

  // Return a function in order to capture and modify the response body:
  return function(body) {
    return body.replace(/foo/g, 'bar')
  }

})
```

## License

(The MIT License)

Copyright (c) 2012 Felix Gnass

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
'Software'), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
