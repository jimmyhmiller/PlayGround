export const MAKE_ADMIN = 'MAKE_ADMIN';

export const makeAdmin = ({ id }) => ({
    type: MAKE_ADMIN,
    id,
})