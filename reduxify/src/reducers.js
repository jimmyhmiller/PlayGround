import { makeAdmin } from './actions';
import { reducers, initial } from './reduxify';

import { setIn } from 'zaphod/compat';


const initialState = {
  users: {
    1: {
      id: 1,
      name: 'Jimmy'
    },
    2: {
      id: 2,
      name: 'Robert',
      admin: true,
    }
  }
}

const setAdmin = (state, { id }) => {
  if (id === undefined || id === null) {
    return state;
  }
  return setIn(state, ['users', id, 'admin'], true)
}


export default reducers(
  initial(initialState),
  makeAdmin.reduce(setAdmin),
)

// selectors

const toArray = obj => Object.keys(obj).map(function (key) { return obj[key]; });

export const getUserById = (state, { id }) => state.users[id] || {};
export const getUserByName = (state, { name }) => toArray(state.users).find(u => u.name === name) || {};

