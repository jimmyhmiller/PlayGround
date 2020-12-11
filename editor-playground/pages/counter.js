import ReduxIde from './index';

const counter = {
  components: {
    Main: {
      code: "<Heading />\n<Button increment={-1} />\n<Button increment={1} />",
      name: "Main",
      type: "component",
    },
    Heading: {
      code: "<h1>My App</h1>",
      name: "Heading",
      type: "component",
      props: [],
    },
    Button: {
      code:
        "<button onClick={_ => Actions.click({ increment: increment || 1 })}>\n{State.clicks} ({increment || 1})\n</button>",
      name: "Button",
      type: "component",
      props: [" increment"],
    },
    State: { code: '{\n "clicks": 0\n}', name: "State", type: "state" },
  },
  actions: {
    click: {
      actionType: "click",
      code:
        "({ clicks, ...state }, { increment }) => ({\n  ...state,\n  clicks: clicks + increment\n})",
    },
  },
};



const Counter = () => {
  return (
    <ReduxIde
      initialComponents={counter.components}
      initialActions={counter.actions} />
   )
}

export default Counter