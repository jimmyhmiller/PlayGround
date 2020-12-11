import ReduxIde from './index';

const todo = {
  components: {
    Main: {
      code: "<Heading />\n<Todos />\n<SubmitTodo />",
      name: "Main",
      type: "component",
    },
    Heading: {
      code: "<h1>Todo</h1>",
      name: "Heading",
      type: "component",
      props: [],
    },
    Todos: {
      code: "<ul>\n  {State.todos.map(({id, item}) => <li>{item}</li>)}\n</ul>",
      name: "Todos",
      type: "component",
      props: [],
    },
    State: {
      code:
        '{\n "todos": [\n  {\n   "id": 1,\n   "item": "stuff"\n  },\n  {\n   "id": 2,\n   "item": "stuff2"\n  }\n ],\n "id": 3\n}',
      name: "State",
      type: "state",
    },
    SubmitTodo: {
      code:
        'const [todo, setTodo] = useState("");\n--\n<input value={todo} onChange={e => setTodo(e.target.value)} />\n<button onClick={_ => Actions.addTodo({ todo })}>Submit</button>',
      name: "SubmitTodo",
      type: "component",
      props: [],
    },
  },
  actions: {
    addTodo: {
      actionType: "addTodo",
      code:
        "({ todos, id, ...state }, { todo }) => ({\n  ...state,\n  id: id + 1,\n  todos: todos.concat([{id, item: todo}])\n\n})",
    },
  },
};

const Todo = () => {
  return (
    <ReduxIde
      initialComponents={todo.components}
      initialActions={todo.actions} />
   )
}

export default Todo