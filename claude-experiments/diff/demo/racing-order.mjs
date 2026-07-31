// Who goes first, alternating.
//
// Both sides of the demo are spawned back to back and then contend for the machine, so
// the process spawned first gets a moment of it to itself. Over a session that is a
// standing advantage for whichever side is always first, and it is free to remove: flip
// the order every race.
//
// The parity is PER KIND of race, which is the part that is easy to get wrong. A single
// shared counter looks right and is not: the production-build handler asks for an order
// twice per press (once for the two builds, once for the reboot afterwards), so a shared
// counter advances by two per press and the build order comes out the same every single
// time — the exact bias this is here to remove.
const parity = new Map();

/// The two sides in the order they should be started for this race, flipping on every
/// call for the same `kind`. `kind` names the race ("build", "boot"), not the sides.
export function racingOrder(kind, sides) {
  const n = parity.get(kind) ?? 0;
  parity.set(kind, n + 1);
  return n % 2 === 0 ? [...sides] : [...sides].reverse();
}

/// Test seam: forget every parity, so a test starts from a known state.
export function resetRacingOrder() {
  parity.clear();
}
