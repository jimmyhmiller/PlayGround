import React from 'react';
import { connect } from 'react-redux';

const overrideMerge = (stateProps, dispatchProps, ownProps) => ({
  ...stateProps,
  ...dispatchProps,
  ...ownProps,
})

const reduxify = Comp => ({ withSelector, withActions, ...rest }) => {
  if (withSelector || withActions) {
    const NewComp = reduxify(connect(withSelector, withActions, overrideMerge)(Comp))
    return <NewComp {...rest} />
  }
  return <Comp {...rest} />
}

export default reduxify;
