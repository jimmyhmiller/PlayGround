import React from 'react';

import reduxify from './reduxify';
import { getUserById } from './selectors';
import { makeAdmin } from './actions';

const User = reduxify(({ name, id, makeAdmin, admin }) => 
  <div>
    <div>Name: {name}</div>
    <div>Admin: {admin ? "Yes" : "No"}</div>
    <button onClick={() => makeAdmin({ id })}>Make Admin</button>
  </div>
)

User.defaultProps = {
  withActions: { makeAdmin },
  withSelector: getUserById,
}


export default User;