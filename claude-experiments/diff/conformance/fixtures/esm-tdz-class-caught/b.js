import { Widget } from "./a.js";
try {
  new Widget();
  console.log("b:constructed");
} catch (error) {
  console.log("b:tdz:" + error.constructor.name);
}
