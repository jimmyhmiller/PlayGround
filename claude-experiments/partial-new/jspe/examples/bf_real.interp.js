// A REAL Brainfuck interpreter, written in jspe's subset (also valid Node JS).
//   - byte cells (mod 256), fixed tape, real I/O (`input`/returned `out` byte arrays)
//   - the 8 commands by ASCII code:  + 43  - 45  > 62  < 60  . 46  , 44
//   - `[ ]` are pre-parsed (by the Node harness) into NESTED blocks, and each loop runs
//     as `while (cell != 0) { ... }` — the standard real-BF structure.
//
// Program form (from the harness's parser): a block is an array whose elements are either
// an op CODE (a number) or a nested loop block (an array). e.g. ,[.,]  ->  [44, [46, 44]]
//
// State is carried in a mutable `ctx` array (jspe is an imperative PE):
//   ctx[0]=tape  ctx[1]=ptr  ctx[2]=inptr  ctx[3]=out  ctx[4]=outlen
// ctx is THREADED through return values (not just shared by reference): jspe's loop
// materialization copies a loop-carried array to a fresh var, so we must pass that copy
// forward explicitly or the accumulated state (e.g. output) is lost after the loop.
function interp(prog, input) {
  var ctx = [newtape(TAPE_SIZE), 0, 0, [], 0];
  ctx = run(prog, ctx, input);
  return ctx[3];
}
function run(items, ctx, input) {
  var i = 0;
  while (i < items.length) {
    var it = items[i];
    if (typeof it === "number") {
      ctx = doop(it, ctx, input);
    } else {
      while (ctx[0][ctx[1]] !== 0) { ctx = run(it, ctx, input); }   // [ ... ]  : while cell != 0
    }
    i = i + 1;
  }
  return ctx;
}
function doop(op, ctx, input) {
  if (op === 43) { ctx[0][ctx[1]] = (ctx[0][ctx[1]] + 1) % 256; }   // +
  if (op === 45) { ctx[0][ctx[1]] = (ctx[0][ctx[1]] + 255) % 256; } // -
  if (op === 62) { ctx[1] = ctx[1] + 1; }                          // >
  if (op === 60) { ctx[1] = ctx[1] - 1; }                          // <
  if (op === 46) { ctx[3][ctx[4]] = ctx[0][ctx[1]]; ctx[4] = ctx[4] + 1; }   // .  (emit byte)
  if (op === 44) {                                                 // ,  (read byte; EOF -> 0)
    if (ctx[2] < input.length) { ctx[0][ctx[1]] = input[ctx[2]]; ctx[2] = ctx[2] + 1; }
    else { ctx[0][ctx[1]] = 0; }
  }
  return ctx;
}
function newtape(n) {
  var t = [];
  var i = 0;
  while (i < n) { t[i] = 0; i = i + 1; }
  return t;
}
