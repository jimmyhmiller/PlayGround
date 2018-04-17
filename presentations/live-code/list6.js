
const isTrue = (bool) => bool;

const not = (bool) => !bool;

const emptyList = () => null

const prepend = (elem, list) => {
    return (message) => {
        if (message === "head") {
            return elem;
        } else if (message === "tail") {
            return list;
        }
    }
}

const head = (list) => list("head");

const tail = (list) => list("tail");

const isEmpty = (list) => list === null;


// functions
// booleans
// properties
// null
// equality
// strings
// if



module.exports = {
    emptyList,
    prepend,
    head,
    tail,
    isTrue,
    not,
    isEmpty,
}