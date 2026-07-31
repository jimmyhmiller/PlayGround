// A DIFFERENT module with an identical primitive value: caching the wrapper by
// VALUE would hand these two modules one shared namespace, which Node does not.
module.exports = 42;
