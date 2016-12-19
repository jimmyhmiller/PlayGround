import { setIn } from 'zaphod/compat';

import { makeAdmin } from './actions';
import { reducers, initial, reducer } from './reduxify';

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
  reducer(makeAdmin, setAdmin),
)