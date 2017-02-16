
const fluentCompose = (f, combinators) => {
  const wrapperFunction = (g) => {
    if (typeof(g) !== 'function') {
      return g;
    }
    const innerFunc = (...args) => g(...args);
    Object.keys(combinators).forEach(k => {
      innerFunc[k] = (...args) => wrapperFunction(combinators[k](g)(...args))
    })
    return innerFunc;
  }
  return wrapperFunction(f);
}

const increment = () => ({
  type: 'INCREMENT'
})

const decrement = () => ({
  type: 'DECREMENT'
})

const run = next => (state, action) => {
  if (action === undefined) {
    action = state;
    state = next();
  }
  return next(state, action)
}

const baseReducer = (state, action) => state;

const initialState = next => init => (state, action) => next(state || init, action);

const reduce = next => (type, f) => (state, action) => {
  if (action && action.type === type) {
    return f(state, action)
  }
  return next(state, action)
}

const reducer = fluentCompose(baseReducer, { initialState, reduce, run })

console.log(
  reducer
    .initialState(0)
    .reduce('INCREMENT', x => x + 1)
    .reduce('DECREMENT', x => x - 1)
    .run(increment())
)


const connect = (mapStateToProps, mapDispatchToProps) => 
                (Comp) => () => Comp({...mapStateToProps({ count: 1 }), ...mapDispatchToProps})

const composeMap = (...mappers) => state => 
  mappers.reduce((props, mapper) => ({
    ...props,
    ...mapper(state),
  }), {})

const emptyMapper = () => ({});

const mapState = next => f => 
  (mapStateToProps=emptyMapper, mapDispatchToProps={}) =>
    next(composeMap(mapStateToProps, f), mapDispatchToProps)

const withActions = next => action => (mapStateToProps=emptyMapper, mapDispatchToProps={}) =>
  next(mapStateToProps, {...mapDispatchToProps, ...action})

const withComponent = next => Comp => (mapStateToProps=emptyMapper, mapDispatchToProps={}) =>
  next(mapStateToProps, mapDispatchToProps)(Comp)

const render = comp => comp();

const reduxify = fluentCompose(connect, { 
  mapState,
  withActions,
  withComponent,
  render,
})

const Comp = (props) => console.log(props);
const OtherComp = (props) => console.log('other', props);

reduxify
  .mapState(state => ({ count: state.count }))
  .mapState(state => ({ y:1 }))
  .withActions({ increment })
  .withComponent(Comp)
  .withActions({ decrement })
  .render()



console.log('\ndone\n')