'lang sweet.js';


import { unwrap, isKeyword } from '@sweet-js/helpers';


export function expandNestedIf(falseBlock) {
  let falseBody = falseBlock.next();
  if (isKeyword(falseBody) && unwrap(falseBody).value === 'if') {
    return falseBody.expand('expr').value;
  } else if (isKeyword(falseBody) && unwrap(falseBody).value === 'return') {
    return falseBlock.expand('expr').value;
  }

}
