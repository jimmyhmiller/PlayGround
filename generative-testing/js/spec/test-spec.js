require('jasmine-check').install()
const { reverse } = require('../index.js')

describe('reverse', () => {
  check.it('reverse reverse', gen.array(gen.int), (xs) => {
    expect(reverse(reverse(xs))).toEqual(xs)
  })
})