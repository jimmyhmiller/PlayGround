// Hand-written smoke test for the lifted node:querystring.
(function () {
const log = [];
const ok = (name, val) => log.push(`OK   ${name}: ${val}`);
const fail = (name, e) => log.push(`FAIL ${name}: ${e.message || e}`);
const check = (name, fn) => { try { ok(name, JSON.stringify(fn())); } catch (e) { fail(name, e); } };

check('typeof querystring',        () => typeof querystring);
check('stringify simple',          () => querystring.stringify({ a: 1, b: 2 }));
check('stringify array',           () => querystring.stringify({ a: [1, 2] }));
check('parse simple',              () => querystring.parse('a=1&b=2'));
check('parse repeated',            () => querystring.parse('a=1&a=2&b=3'));
check('parse %-encoded',           () => querystring.parse('a=hello%20world'));
check('parse +-encoded',           () => querystring.parse('a=hello+world'));
check('escape',                    () => querystring.escape('hello world'));
check('escape special',            () => querystring.escape('foo=bar&baz'));
check('unescape',                  () => querystring.unescape('hello%20world'));
check('unescape +',                () => querystring.unescape('hello+world'));
check('roundtrip',                 () => {
  const obj = { name: 'foo bar', tags: ['a', 'b'], n: 42 };
  const s = querystring.stringify(obj);
  const back = querystring.parse(s);
  return { s, back };
});

return log.join('\n');
})()
