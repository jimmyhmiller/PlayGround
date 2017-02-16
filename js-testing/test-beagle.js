// data Action = Increment | Decrement

// data Maybe = None | Just(a)

// data Person = Customer({ :id })
//             | Employee({ :id, :position })


// fn get-customer-id {
//     Customer({ id }) => Just(id)
//     Employee(_) => None
// }

// fn counter {
//     (state, Increment) => state + 1
//     (state, Decrement) => state - 1
//     (state, _) => state
// }

const Increment = {
    type: 'Action/Increment'
}

const Decrement = {
    type: 'Action/Decrement'
}

const None = {
    type: 'Maybe/None'
}

const Just = (a) => ({
    type: 'Maybe/Just',
    args: [a]
})

const Customer = ({ id }) => ({
    type: 'Person/Customer',
    args: [{ id }]
})

const Employee = ({ id, position }) => ({
    type: 'Person/Employee',
    args: [{ id, position }]
})

const getCustomerId = (arg) => {
    if (arg && arg.type === 'Person/Customer') {
        const id = arg.args[0].id;
        return Just(id);
    } else if (arg && arg.type === 'Person/Employee') {
        return None;
    } else {
        throw Error(`No match found for getCustomerId given arguments ${arg}`)
    }
}

const counter = (state, arg) => {
    if (arg && arg.type === 'Action/Increment') {
        return state + 1;
    } else if (arg && arg.type === 'Action/Decrement') {
        return state - 1;
    } else {
        return state;
    }
}
console.log(getCustomerId(Customer({ id: 1 })))
console.log(getCustomerId(Employee({ id: 1, position: 'test' })))



console.log(counter(1, undefined))

console.log('\n\n')


