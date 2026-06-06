module.exports = [
  {name: 'Num zero', fn: () => {
    const { AB } = require('../contracts.js');
    const { absTruthy } = require('../state.js');
    if (absTruthy(AB.Num(0)) !== false) throw new Error('Num 0 should be false');
  }},
  {name: 'Num non-zero', fn: () => {
    const { AB } = require('../contracts.js');
    const { absTruthy } = require('../state.js');
    if (absTruthy(AB.Num(5)) !== true) throw new Error('Num 5 should be true');
  }},
  {name: 'Empty string', fn: () => {
    const { AB } = require('../contracts.js');
    const { absTruthy } = require('../state.js');
    if (absTruthy(AB.Str('')) !== false) throw new Error('Empty string should be false');
  }},
  {name: 'Non-empty string', fn: () => {
    const { AB } = require('../contracts.js');
    const { absTruthy } = require('../state.js');
    if (absTruthy(AB.Str('hello')) !== true) throw new Error('Non-empty string should be true');
  }},
  {name: 'Bool true', fn: () => {
    const { AB } = require('../contracts.js');
    const { absTruthy } = require('../state.js');
    if (absTruthy(AB.Bool(true)) !== true) throw new Error('Bool true should be true');
  }},
  {name: 'Bool false', fn: () => {
    const { AB } = require('../contracts.js');
    const { absTruthy } = require('../state.js');
    if (absTruthy(AB.Bool(false)) !== false) throw new Error('Bool false should be false');
  }},
  {name: 'Undef', fn: () => {
    const { AB } = require('../contracts.js');
    const { absTruthy } = require('../state.js');
    if (absTruthy(AB.Undef()) !== false) throw new Error('Undef should be false');
  }},
  {name: 'Null', fn: () => {
    const { AB } = require('../contracts.js');
    const { absTruthy } = require('../state.js');
    if (absTruthy(AB.Null()) !== false) throw new Error('Null should be false');
  }},
  {name: 'Ref', fn: () => {
    const { AB } = require('../contracts.js');
    const { absTruthy } = require('../state.js');
    if (absTruthy(AB.Ref(42)) !== true) throw new Error('Ref should be true');
  }},
  {name: 'Dyn', fn: () => {
    const { AB, RE } = require('../contracts.js');
    const { absTruthy } = require('../state.js');
    if (absTruthy(AB.Dyn(RE.Num(0))) !== null) throw new Error('Dyn should be null');
  }}
];