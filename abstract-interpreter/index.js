const acorn = require("acorn");
const util = require('util');
const zaphod = require("zaphod/compat");
const { threadFirst, fluentCompose } = require('fluent-compose');

const transform = fluentCompose(threadFirst(zaphod))

// const oldLog = console.log.bind(console);
// console.log = (x) => {
//   oldLog(util.inspect(x, {showHidden: false, depth: null}))
// }

const NEXT = "NEXT";


const code = `
let x = 3;
let y;

if (x > 3) {
  y = 3;
}
`

const DONE = "DONE";

const processCode = ({ expr, state }) => {
  switch (expr.type) {
    case "Program": {
      return { expr, state };
    }
    case "VariableDeclarator": {
      return {
        expr,
        state: {
          ...state,
          [expr.id.name]: expr.init && expr.init.value
        },
        action: DONE
      }
    }
    default: {
      return {expr, state: {...state, nodes: (state.nodes || []).concat([expr.type])}}
    }
  }
}




const process = (f, context) => {
  const { expr, state } = context;
  switch (expr.type) {
    case "Program": {
      return expr.body.reduce((context, expr) => {
        const { expr: newExpr, state: newState } = process(f, {state: context.state, expr})
        return transform
          .set("state", newState)
          .updateIn(["expr", "body"], body => body.concat(newExpr))
          (context)
      }, transform.setIn(["expr", "body"], [])(context))
    }
    default: {
      return f(context);
    }
  }
}



console.log(process(processCode, {expr: acorn.parse(code), state: {}}))












