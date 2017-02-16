import React from 'react';
import ReactDOM from 'react-dom';
import App from './App';
import './index.css';
import { createStore, applyMiddleware } from 'redux'
import { Provider } from 'react-redux';


const logger = store => next => action => {
  console.log('dispatching', action)
  let result = next(action)
  console.log('next state', store.getState())
  return result
}

const TAKE = 'TAKE';
const PUT = 'PUT';

const take = actionType => ({
  type: TAKE,
  actionType
})

const put = action => ({
  type: PUT,
  action
})

function* saga() {
  yield put({
    type: 'ACTION2',
    y: 1
  })
  const action = yield take('ACTION');
  console.log(action)
  const action2 = yield take('ACTION2');
  console.log('got y', action2.y)
  const action3 = yield take('ACTION');
  console.log('end!')
}

const log = (...args) => {
  console.log(...args);
  return args[args.length-1];
}

const consume = (effect) => ({
  ...effect,
  consumed: true,
})

const next = (saga, store, value) => {
  const effect = saga.next(value);
  store.dispatch({
    type: 'SAGA',
    effect: effect.value,
  })
  return effect.value;
}

const getEffect = store => store.getState().saga.effect;

const processTake = (saga, store, action, effect) => {
  if (effect.actionType === action.type) {
    return consume(next(saga, store, action));
  }
  return effect;
}

const processPut = (saga, store, action, effect) => {
  next(saga, store, action);
  store.dispatch(effect.action);
  return getEffect(store);
}

const processEffect = (saga, store, action, effect) => {
  if (!effect) {
    return effect;
  }
  if (effect.type === TAKE) {
    return processTake(saga, store, action, effect)
  }
  if (effect.type === PUT) {
    return processPut(saga, store, action, effect)
  }
  return effect;
}

const processSaga = (saga, store, action) => {
  const effect = getEffect(store)
  const newEffect = processEffect(saga, store, action, effect);
  if (effect !== newEffect) {
    processSaga(saga, store, effect.consumed ? {} : action)
  }
}

const runSaga = saga => store => {
  next(saga, store);
  return next => action => {
    const result = next(action) // hit reducers
    if (action.type !== 'SAGA') {
      processSaga(saga, store, action);
    }
    return result;
  }
}

let reducer = (state={saga: {}, actions: []}, action) => {
  if (action.type === 'SAGA') {
    return {
      ...state,
      saga: {
        effect: action.effect
      },
    }
  }
  return {
    ...state,
    actions: state.actions.concat(action),
  }
}
let store = createStore(
  reducer,
  // applyMiddleware() tells createStore() how to handle middleware
  applyMiddleware(runSaga(saga()))
)


ReactDOM.render(
  <Provider store={store}>
    <App />
  </Provider>,
  document.getElementById('root')
);
