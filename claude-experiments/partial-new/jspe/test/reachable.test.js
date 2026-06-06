module.exports = [
  {
    name: 'reachable marks a single object',
    fn: () => {
      const { reachable } = require('../state.js');
      const heap = new Map();
      heap.set(1, { tag: 'Object', fields: [] });
      const outSet = new Set();
      reachable(heap, 1, outSet);
      if (!outSet.has(1)) throw new Error('should mark addr 1');
      if (outSet.size !== 1) throw new Error('should only mark one');
    }
  },
  {
    name: 'reachable traverses object fields',
    fn: () => {
      const { reachable } = require('../state.js');
      const heap = new Map();
      heap.set(1, { tag: 'Object', fields: [['x', { tag: 'Ref', addr: 2 }]] });
      heap.set(2, { tag: 'Object', fields: [] });
      const outSet = new Set();
      reachable(heap, 1, outSet);
      if (!outSet.has(1) || !outSet.has(2)) throw new Error('should mark both');
    }
  },
  {
    name: 'reachable traverses array elements',
    fn: () => {
      const { reachable } = require('../state.js');
      const heap = new Map();
      heap.set(1, { tag: 'Array', elems: [{ tag: 'Ref', addr: 2 }, { tag: 'Num', n: 5 }] });
      heap.set(2, { tag: 'Object', fields: [] });
      const outSet = new Set();
      reachable(heap, 1, outSet);
      if (!outSet.has(1) || !outSet.has(2)) throw new Error('should mark both');
    }
  },
  {
    name: 'reachable traverses closure captured',
    fn: () => {
      const { reachable } = require('../state.js');
      const heap = new Map();
      heap.set(1, { tag: 'Closure', fid: 0, captured: [{ tag: 'Ref', addr: 2 }] });
      heap.set(2, { tag: 'Object', fields: [] });
      const outSet = new Set();
      reachable(heap, 1, outSet);
      if (!outSet.has(1) || !outSet.has(2)) throw new Error('should mark both');
    }
  },
  {
    name: 'reachable handles cycles',
    fn: () => {
      const { reachable } = require('../state.js');
      const heap = new Map();
      heap.set(1, { tag: 'Object', fields: [['self', { tag: 'Ref', addr: 1 }]] });
      const outSet = new Set();
      reachable(heap, 1, outSet);
      if (!outSet.has(1)) throw new Error('should mark addr 1');
      if (outSet.size !== 1) throw new Error('should not infinite loop');
    }
  }
];