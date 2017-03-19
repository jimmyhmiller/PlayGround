"use strict";
console.log('new')
const im = require('immutable')
const List = im.List;
const fromJS = im.fromJS;

const map = fromJS({name: {first: 'jimmy', last: 'Miller'}, age: 2})
const fields = (fs, map, acc={}) => {
    if (fs.isEmpty()) {
        return acc;
    } else {
        let value = fs.first()
        acc[value] = map.getIn(value.split('.'));
        return fields(fs.pop(), map, acc);
    }
}

console.log('here')
const {name, age} = fields(List.of('name.first', 'age'), map);

console.log(map.getIn('name.first'.split('.')))

console.log(name)