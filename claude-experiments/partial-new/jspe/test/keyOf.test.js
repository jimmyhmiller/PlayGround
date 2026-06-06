module.exports = [
  {
    name: 'keyOf empty state',
    fn: () => {
      const { keyOf, alloc } = require('../state.js');
      const { AB } = require('../contracts.js');
      const state = { frames: [], heap: new Map(), nextAddr: 0, pendingJoins: [], handlers: [] };
      const k = keyOf(state);
      if (typeof k !== 'string' || k.length === 0) throw new Error('key should be non-empty string');
    }
  },
  {
    name: 'keyOf two equal states produce same key',
    fn: () => {
      const { keyOf, alloc } = require('../state.js');
      const { AB } = require('../contracts.js');
      const state1 = { frames: [{ pc: 0, func: 0, locals: [AB.Num(1)], ostack: [] }], heap: new Map(), nextAddr: 0, pendingJoins: [], handlers: [] };
      const state2 = { frames: [{ pc: 0, func: 0, locals: [AB.Num(1)], ostack: [] }], heap: new Map(), nextAddr: 0, pendingJoins: [], handlers: [] };
      if (keyOf(state1) !== keyOf(state2)) throw new Error('equal states should have same key');
    }
  },
  {
    name: 'keyOf different states produce different keys',
    fn: () => {
      const { keyOf } = require('../state.js');
      const { AB } = require('../contracts.js');
      const state1 = { frames: [{ pc: 0, func: 0, locals: [AB.Num(1)], ostack: [] }], heap: new Map(), nextAddr: 0, pendingJoins: [], handlers: [] };
      const state2 = { frames: [{ pc: 0, func: 0, locals: [AB.Num(2)], ostack: [] }], heap: new Map(), nextAddr: 0, pendingJoins: [], handlers: [] };
      if (keyOf(state1) === keyOf(state2)) throw new Error('different states should have different keys');
    }
  },
  {
    name: 'keyOf with heap object',
    fn: () => {
      const { keyOf, alloc } = require('../state.js');
      const { AB } = require('../contracts.js');
      const state = { frames: [{ pc: 0, func: 0, locals: [AB.Ref(0)], ostack: [] }], heap: new Map(), nextAddr: 0, pendingJoins: [], handlers: [] };
      alloc(state, { tag: 'Object', fields: [['x', AB.Num(42)]] });
      const k = keyOf(state);
      if (typeof k !== 'string') throw new Error('key should be string');
    }
  }
];