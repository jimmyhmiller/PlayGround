const connect = (mapStateToProps, mapDispatchToProps) => 
                (Comp) => () => console.log({...mapStateToProps({ count: 1 }), ...mapDispatchToProps})

const composeMap = (...mappers) => state => 
  mappers.reduce((props, mapper) => ({
    ...props,
    ...mapper(state),
  }), {})


const emptyMapper = () => ({});

const mapState = f => (mapStateToProps=emptyMapper, mapDispatchToProps={}) => 
  [composeMap(mapStateToProps, f), mapDispatchToProps]

const withActions = action => (mapStateToProps=emptyMapper, mapDispatchToProps={}) =>
  [mapStateToProps, {...mapDispatchToProps, ...action}]

const wrap = (wrapperFunction, comb, args) => 
             (...combArgs) => 
             wrapperFunction(...comb(...combArgs)(...args))

const fluentCompose = (f, combinators) => {
  const wrapperFunction = (...args) => {
    const innerFunc = f(...args);
    Object.keys(combinators).forEach(k => {
      innerFunc[k] = wrap(wrapperFunction, combinators[k], args)
    })
    return innerFunc;
  }

  return wrapperFunction();
}

const connector = fluentCompose(connect, { mapState, withActions });

const reduxify2 = fluentCompose(connector, { withComp: x => x })

const reduxify = (Comp) => {
  if (Comp) {
    return fluentCompose((...args) => connect(...args)(Comp), { mapState, withActions })
  }
  return fluentCompose(connect, { mapState, withActions })
}

const Comp = (props) => console.log(props);


const increment = ({
  type: 'INCREMENT'
})

reduxify(Comp)
  .mapState(state => ({ count: state.count }))
  .withActions({ increment })


reduxify()
  .mapState(state => ({ count: state.count }))
  .withActions({ increment })(Comp)()

console.log('\ndone\n')



