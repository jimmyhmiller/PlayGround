const toArray = obj => Object.keys(obj).map(function (key) { return obj[key]; });

export const getUserById = (state, { id }) => state.users[id] || {};
export const getUserByName = (state, { name }) => toArray(state.users).find(u => u.name === name) || {};
