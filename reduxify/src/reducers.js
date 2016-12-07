import { MAKE_ADMIN } from './actions';

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

const makeAdmin = (state, { id }) => {
  if (id === undefined || id === null) {
    return state;
  }
  return {
    ...state,
    users: {
      ...state.users,
      [id]: {
        ...state.users[id],
        admin: true
      }  
    }
  }
}


const toArray = obj => Object.keys(obj).map(function (key) { return obj[key]; });

export const getUserById = (state, { id }) => state.users[id] || {};
export const getUserByName = (state, { name }) => toArray(state.users).find(u => u.name === name) || {};

export default (state=initialState, action) => {
  switch (action.type) {
    case MAKE_ADMIN:
      return makeAdmin(state, action)
    default:
      return state
  }
}