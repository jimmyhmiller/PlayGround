import { add } from "./utils.js";

export function greet(name) {
  const num = add(1, 2);
  return `Hello, ${name}! The answer is ${num}`;
}

export function farewell(name) {
  return `Goodbye, ${name}!`;
}
