const Counter = ({ increment, decrement, count }) =>({ 
  increment, 
  decrement, 
  count
})

const connect = (mapStateToProps, mapDispatchToProps) => (Comp) => console.log(mapStateToProps, mapDispatchToProps)


const composeMapState = (...mappers) => state => 
  mappers.reduce((props, mapper) => ({
    ...props,
    ...mapper(state),
  }), {})


const reduxify = (Comp, mapStateToProps=[], mapDispatchToProps={}) => {
  const innerComp = connect(composeMapState(...mapStateToProps), mapDispatchToProps)(Comp);
  innerComp.withAction = actions => reduxify(
    Comp,
        mapStateToProps,
    {...mapDispatchToProps, ...actions},
  )
  innerComp.mapState = mapper => reduxify(
    Comp,
    mapStateToProps.concat(mapper),
    mapDispatchToProps,
  )
  return innerComp;
} 


reduxify(Counter)
  .withAction({ increment })
  .withAction({ decrement })
  .mapState(state => state.count)

