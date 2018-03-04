import { unwrap, isKeyword } from '@sweet-js/helpers' for syntax;





// The `syntax` keyword is used to create and name new macros.
syntax if = function (ctx) {
  let pred = ctx.next().value;
  let trueBlock = ctx.contextify(ctx.next().value);
  let _trueReturn = trueBlock.next()
  let trueCase = trueBlock.next().value;
  let _else = ctx.next();
  let falseBlock = ctx.contextify(ctx.next().value)

  let _return = falseBlock.next()
  let falseValue = falseBlock.next().value
  

  return #`IF(${pred}, () => ${trueCase}, () => ${falseValue})()`;
}

TRUE = (t, f) => t;
FALSE = (t, f) => f;
IF = (pred, t, f) => pred(t, f); 


function thing () {
    if (TRUE) {
        return 3;
    } else {
      return 2;
    }
}

