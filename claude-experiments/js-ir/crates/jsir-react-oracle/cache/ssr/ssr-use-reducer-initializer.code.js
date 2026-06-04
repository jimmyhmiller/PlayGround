// @outputMode:"ssr"

import { useReducer } from "react";

const initializer = (x) => {
  return x;
};

function Component() {
  const state = initializer(0);

  return <input value={state} />;
}
