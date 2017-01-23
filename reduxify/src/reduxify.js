import React from 'react';
import { connect } from 'react-redux';
import { keys } from 'zaphod/compat';

const difference = function(setA, setB) {
    const temp = new Set(setA);
    for (const elem of setB) {
        temp.delete(elem);
    }
    return temp;
}

const printWarningMessage = (schema, values, type) => {
  const schemaSet = new Set(keys(schema));
  const valuesSet = new Set(keys(values));
  const missingKeys = difference(schemaSet, valuesSet);
  const extraKeys = difference(valuesSet, schemaSet);
  if (extraKeys.size > 0 && missingKeys.size > 0) {
      console.warn(`Action of type ${type} is missing the following key(s): [${Array.from(missingKeys.values()).join(", ")}]\nIt also has the following extra key(s): [${Array.from(extraKeys.values()).join(", ")}]\nThe action creator was given the following object:\n${JSON.stringify(values, null, '  ')}`)
  }
  else if (missingKeys.size > 0) {
      console.warn(`Action of type ${type} is missing the following key(s): [${Array.from(missingKeys.values()).join(", ")}]\nThe action creator was given the following object:\n${JSON.stringify(values, null, '  ')}`)
  }
  else if (extraKeys.size > 0) {
      console.warn(`Action of type ${type} has the following extra key(s): [${Array.from(extraKeys.values()).join(", ")}]\nThe action creator was given the following object:\n${JSON.stringify(values, null, '  ')}`)
  }
}

export const reducers = (red, ...reds) => (state, action) => 
  reds.reduce((state, f) => f(state, action), red(state, action))

export const initial = initialState => state => state || initialState;

export const reducer = (actionCreatorOrType, f) => (state, action) => {
  const type = typeof actionCreatorOrType === 'string' ? actionCreatorOrType : actionCreatorOrType.type;
  if (action && action.type === type) {
    return f(state, action)
  } 
  return state
}

export const action = ({ type, ...schema }) => {
    const actionCreator = (values) => {
        printWarningMessage(schema, values, type);
        return {
            type,
            ...values,
        }
    }
    actionCreator.type = type;
    return actionCreator;
}

const overrideMerge = (stateProps, dispatchProps, ownProps) => ({
  ...stateProps,
  ...dispatchProps,
  ...ownProps,
})

const reduxify = Comp => ({ withSelector, withActions, noConnect, ...rest }) => {
  if ((withSelector || withActions) && !noConnect) {
    const NewComp = reduxify(connect(withSelector, withActions, overrideMerge)(Comp))
    return <NewComp {...rest} />
  }
  return <Comp {...rest} />
}

export default reduxify;