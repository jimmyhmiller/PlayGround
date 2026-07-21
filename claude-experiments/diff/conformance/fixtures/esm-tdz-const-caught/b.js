import { value } from "./a.js";
try {
  console.log("b:sees:" + value);
} catch (error) {
  console.log("b:tdz:" + error.constructor.name);
}
