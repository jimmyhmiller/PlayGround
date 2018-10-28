import React, { Component, useState, useEffect } from 'react';
import { useDispatch, useSelector, useActions, addTodo as addTodoAction } from './ReduxExperiment';

// Notes to self
// useMemo doesn't memo. It just does a diff with previous value.
// All it does is prevent rerenders when nothing has changed,
// it doesn't save the result of old values

const useCounter = (initial, increment) => {
  const [count, setCount] = useState(initial);
  return [count, () => setCount(count + increment)]
}

const useFetchData = (initial, endpoint) => {
  const [data, setData] = useState(initial);

  useEffect(() => {
    fetch(endpoint)
      .then(resp => {
        if (resp.status !== 200) {
          return []
        }
        return resp.json()
      })
      .then(data => setData(data))
  }, [endpoint])

  return data;
}

const useFormInput = (initial) => {
  const [value, setValue] = useState(initial);
  return {
    value,
    onChange: (e) => setValue(e.target.value),
  }
}

const Todo = ({ text }) => <div>{text}</div>

const Gist = ({ files }) =>
  <div>{Object.keys(files)[0]}</div>


const App = (props) => {
  const [count, increment] = useCounter(0, 1);
  const todos = useSelector(x => x) || [];
  let { addTodo } = useActions({ addTodo: addTodoAction });
  const user = useFormInput('jimmyhmiller');
  const todo = useFormInput('');
  const gists = useFetchData([], `https://api.github.com/users/${user.value}/gists`);

  return (
    <>
      <div>
        <h3>Gists</h3>
        {gists.map(gist => <Gist key={gist.id} {...gist} />)}
        <div>
          <input {...user} />
        </div>
      </div>

      <div>
        <h3>Counter</h3>
        <button
          onClick={increment}>
          Count: {count}
        </button>
      </div>

      <div>
        <h3>Todos</h3>
        {todos.map(todo => <Todo key={todo.text} {...todo} />)}
        <div>
          <input {...todo} />
          <button
            onClick={() => addTodo(todo.value)}>
            Add Todo
          </button>
        </div>
      </div>
    </>
  )
}

export default App;
