module.exports = [
  { name: 'Num', fn: () => {
    const { absToRExpr } = require('../state.js');
    const { AB, RE } = require('../contracts.js');
    const abs = AB.Num(42);
    const result = absToRExpr(abs);
    if (result.tag !== 'Num' || result.n !== 42) throw new Error('Num failed');
  }},
  { name: 'Str', fn: () => {
    const { absToRExpr } = require('../state.js');
    const { AB, RE } = require('../contracts.js');
    const abs = AB.Str('hello');
    const result = absToRExpr(abs);
    if (result.tag !== 'Str' || result.s !== 'hello') throw new Error('Str failed');
  }},
  { name: 'Bool', fn: () => {
    const { absToRExpr } = require('../state.js');
    const { AB, RE } = require('../contracts.js');
    const abs = AB.Bool(true);
    const result = absToRExpr(abs);
    if (result.tag !== 'Bool' || result.b !== true) throw new Error('Bool failed');
  }},
  { name: 'Undef', fn: () => {
    const { absToRExpr } = require('../state.js');
    const { AB, RE } = require('../contracts.js');
    const abs = AB.Undef();
    const result = absToRExpr(abs);
    if (result.tag !== 'Undef') throw new Error('Undef failed');
  }},
  { name: 'Null', fn: () => {
    const { absToRExpr } = require('../state.js');
    const { AB, RE } = require('../contracts.js');
    const abs = AB.Null();
    const result = absToRExpr(abs);
    if (result.tag !== 'Null') throw new Error('Null failed');
  }},
  { name: 'Dyn', fn: () => {
    const { absToRExpr } = require('../state.js');
    const { AB, RE } = require('../contracts.js');
    const expr = RE.Num(1);
    const abs = AB.Dyn(expr);
    const result = absToRExpr(abs);
    if (result !== expr) throw new Error('Dyn failed');
  }},
  { name: 'Ref throws', fn: () => {
    const { absToRExpr } = require('../state.js');
    const { AB } = require('../contracts.js');
    const abs = AB.Ref(0);
    let threw = false;
    try { absToRExpr(abs); } catch (e) { threw = true; }
    if (!threw) throw new Error('Ref should throw');
  }}
];