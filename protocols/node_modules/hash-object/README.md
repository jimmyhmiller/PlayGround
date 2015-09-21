# hash-object

[![Build Status](https://secure.travis-ci.org/davidcaseria/hash-object.png?branch=master)](http://travis-ci.org/davidcaseria/hash-object)

A Node.js module that hashes JavaScript objects.

## Getting Started

Install the module with: `npm install hash-object`


## Documentation

### hashObject(obj, [opts])

```js
var hashObject = require('hash-object');
```

- `obj`: the object to be hashed
- `opts`: options for the hashing function
  - `algorithm`: dependent on the available algorithms supported by the version of OpenSSL on the platform. Examples are 'sha1' (default), 'md5', 'sha256', 'sha512', etc.


## Examples

### Simple Compare

```js
var hash1 = hashObject({
    a: 'b',
    c: 'd',
    e: 'f'
});
var hash2 = hashObject({
    a: 'b',
    e: 'f',
    c: 'd'
});

// hash1 === hash2

```

### Complex Compare

```js
var hash3 = hashObject({
    a: 'b',
    b: ['c', 'd'],
    c: {
        d: 'e',
        f: {
            g: 'h'
        }
    }
});
var hash4 = hashObject({
    a: 'b',
    c: {
        f: {
            g: 'h'
        },
        d: 'e'
    },
    b: ['c', 'd']
});

// hash3 === hash4

```


## Contributing

In lieu of a formal styleguide, take care to maintain the existing coding style. Add unit tests for any new or changed functionality. Lint and test your code using [Grunt](http://gruntjs.com).


## License

Copyright (c) 2014 David Caseria  
Licensed under the MIT license.
