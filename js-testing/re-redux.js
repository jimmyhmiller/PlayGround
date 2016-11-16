

connect = actions => actionSelector => comp => (props) => comp(Object.assign({}, actionSelector(actions), props));
connected = connect({increment: () => ({type: 'INCREMENT'}), decrement: () => ({type: 'DECREMENT'})});


withActions = actionSelector => connected(actionSelector)
enhance =
counter = enhance(({increment, decrement, a}) => [increment(), decrement(), a]);

connectedCounter = withActions(({increment, decrement }) => ({increment, decrement}))(
  counter
)

connectedCounter({a: 1})
