// Who goes first must actually alternate.
//
//   node demo/racing-order.test.mjs
//
// This exists because the first version of the fix was wrong in a way that looked
// right: one shared parity counter, and a production-build press that asks for an
// order twice (the two builds, then the reboot afterwards). The counter advanced by
// two per press, so the build order came out `dp` first every single time — the bias
// the change was made to remove, preserved by the change itself.
import { racingOrder, resetRacingOrder } from "./racing-order.mjs";

const SIDES = [{ key: "dp" }, { key: "tp" }];
const keys = (kind) => racingOrder(kind, SIDES).map((s) => s.key);

let failures = 0;
function check(what, cond, detail) {
  if (cond) console.log(`  ok   ${what}`);
  else {
    failures++;
    console.log(`  FAIL ${what}${detail ? `\n       ${detail}` : ""}`);
  }
}

resetRacingOrder();
const four = [keys("build"), keys("build"), keys("build"), keys("build")];
check(
  "four races of one kind alternate",
  JSON.stringify(four) === JSON.stringify([["dp", "tp"], ["tp", "dp"], ["dp", "tp"], ["tp", "dp"]]),
  JSON.stringify(four),
);

// The regression: a build press consumes one "build" order and one "boot" order.
resetRacingOrder();
const builds = [];
for (let press = 0; press < 4; press++) {
  builds.push(keys("build"));
  keys("boot"); // the reboot after the build
}
check(
  "a build order still alternates when each press also asks for a boot order",
  JSON.stringify(builds) === JSON.stringify([["dp", "tp"], ["tp", "dp"], ["dp", "tp"], ["tp", "dp"]]),
  JSON.stringify(builds),
);

resetRacingOrder();
keys("build");
check("kinds do not share a counter", JSON.stringify(keys("boot")) === JSON.stringify(["dp", "tp"]), JSON.stringify(keys("boot")));

// A restart of ONE side is a valid request; it must not lose the side or duplicate it.
resetRacingOrder();
const single = racingOrder("boot", [{ key: "tp" }]).map((s) => s.key);
check("a one-side race returns exactly that side", JSON.stringify(single) === JSON.stringify(["tp"]), JSON.stringify(single));

resetRacingOrder();
const input = [...SIDES];
racingOrder("boot", input);
racingOrder("boot", input);
check(
  "the caller's array is never reordered in place",
  JSON.stringify(input.map((s) => s.key)) === JSON.stringify(["dp", "tp"]),
  JSON.stringify(input.map((s) => s.key)),
);

// Over many races each side leads half the time, which is the whole point.
resetRacingOrder();
let dpFirst = 0;
const RACES = 100;
for (let i = 0; i < RACES; i++) if (keys("build")[0] === "dp") dpFirst++;
check(`each side leads half of ${RACES} races`, dpFirst === RACES / 2, `dp led ${dpFirst}`);

console.log(failures ? `\n${failures} check(s) failed` : "\nall checks passed");
process.exit(failures ? 1 : 0);
