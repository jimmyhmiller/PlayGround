module.exports = [
  {
    name: 'gc removes unreachable object',
    fn: () => {
      const { AB } = require('../contracts.js');
      const { alloc, gc } = require('../state.js');
      const state = { frames: [{ pc: 0, func: 0, locals: [], ostack: [] }], heap: new Map(), nextAddr: 0, pendingJoins: [], handlers: [] };
      const addr = alloc(state, { tag: 'Object', fields: [] });
      gc(state);
      if (state.heap.has(addr)) throw new Error('should have been removed');
    }
  },
  {
    name: 'gc keeps reachable object',
    fn: () => {
      const { AB } = require('../contracts.js');
      const { alloc, gc } = require('../state.js');
      const state = { frames: [{ pc: 0, func: 0, locals: [AB.Ref(0)], ostack: [] }], heap: new Map(), nextAddr: 0, pendingJoins: [], handlers: [] };
      const addr = alloc(state, { tag: 'Object', fields: [] });
      gc(state);
      if (!state.heap.has(addr)) throw new Error('should be kept');
    }
  },
  {
    name: 'gc keeps object reachable via chain',
    fn: () => {
      const { AB } = require('../contracts.js');
      const { alloc, gc } = require('../state.js');
      const state = { frames: [{ pc: 0, func: 0, locals: [AB.Ref(0)], ostack: [] }], heap: new Map(), nextAddr: 0, pendingJoins: [], handlers: [] };
      const inner = alloc(state, { tag: 'Object', fields: [] });
      const outer = alloc(state, { tag: 'Object', fields: [['x', AB.Ref(inner)]] });
      // adjust locals to point to outer
      state.frames[0].locals = [AB.Ref(outer)];
      gc(state);
      if (!state.heap.has(outer)) throw new Error('outer should be kept');
      if (!state.heap.has(inner)) throw new Error('inner should be kept');
    }
  }
];