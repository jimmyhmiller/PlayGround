import React from 'react';
import ReactDOM from 'react-dom';
import { createStore, combineReducers, compose, applyMiddleware } from 'redux'
import { Provider } from 'react-redux'
import createSagaMiddleware from 'redux-saga'

import './index.css';
import App from './react/App';
import * as serviceWorker from './serviceWorker';
import { editorReducer } from './react/reducers';
import sagas from './react/sagas'


const composeEnhancers = window.__REDUX_DEVTOOLS_EXTENSION_COMPOSE__ || compose;

// create the saga middleware
const sagaMiddleware = createSagaMiddleware()


const store = createStore(
  combineReducers({
    editors: editorReducer
  }),
  composeEnhancers(applyMiddleware(sagaMiddleware))
);

ReactDOM.render(
  <Provider store={store}>
   <App />
  </Provider>, document.getElementById('root'));

// If you want your app to work offline and load faster, you can change
// unregister() to register() below. Note this comes with some pitfalls.
// Learn more about service workers: http://bit.ly/CRA-PWA
serviceWorker.unregister();


sagaMiddleware.run(sagas)





// IPC Example
// 
// useEffect(() => {
//   ipcRenderer.send(channels.APP_INFO);
// 
//   ipcRenderer.on(channels.APP_INFO, (event, { appName, appVersion }) => {
//     // setAppVersion({ appName, appVersion });
//   })
//   return () => ipcRenderer.removeAllListeners(channels.APP_INFO);
// }, [])