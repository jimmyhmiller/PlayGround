import Head from 'next/head'
import { useState, useReducer, useEffect } from 'react';

const colors = {
  "let": "#2aa198",
  "identifier": "#859900",
  "integer": "#b58900",
  "unknown": "#9c9c9c",
}
const defaultColor = "#000000";

// $yellow:    #b58900;
// $orange:    #cb4b16;
// $red:       #dc322f;
// $magenta:   #d33682;
// $violet:    #6c71c4;
// $blue:      #268bd2;
// $cyan:      #2aa198;
// $green:     #859900;

const Node = ({ text, type }) => {
  const hasBorder = type !== "whitespace" && type !== "newline" && type !== "cursor";
  const color = colors[type] || defaultColor;
  return (
    <span 
      style={{
        color: color,
        padding: hasBorder ? 3 : undefined,
        border: hasBorder ? `solid 1px ${color}` : undefined,
      }}
      className={type === "cursor" ? "cursor" : undefined}>
    {text}
    </span>
  )
}

const useKeyPress = (handler) => {
   useEffect(() => {
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler)
  }, [])
}

const inferType = (nodes, currentNode) => {
  if (currentNode.text === "let" || currentNode.text === "const") {
    return "let"
  } else if (!isNaN(parseInt(currentNode.text, 10))) {
    return "integer"
  } else if (currentNode.text.match(/[A-Za-z=\-+&|\/\?][A-Za-z=\-+&|\/\?0-9]?/)) {
    return "identifier"
  }
  else {
    return "unknown"
  }
}

const reducer = (state, { type, ...action }) => {
  console.log({ type, ...action })
  switch (type) {
    case "press": {
      switch (action.key) {
        case "Shift":
        case "Meta":
        case "Control":
        case "Alt":
        case "Escape": {
          return state
        }
        case " ": {
          return {
            ...state,
            id: state.id + 1,
            nodes: state.nodes
              .filter(x => x.type !=="cursor")
              .concat([{ type: "whitespace", text: " ", id: state.id }, 
                       { type: "cursor", id: -1}])
          }
        }
        case "Enter": {
          return {
            ...state,
            id: state.id + 1,
            nodes: state.nodes
              .filter(x => x.type !== "cursor")
              .concat([{ type: "newline", text: "\n", id: state.id }, 
                       { type: "cursor", id: -1}])
          }
        }
        case "Backspace": {
          action.event.preventDefault();
          // We have a cursor so it is 1 not 0
          if (state.nodes.length === 1) {
            break;
          }
          const newNodes = [...state.nodes];
          newNodes.pop(); // get rid of cursor in a dumb way need to actually care about location
          const node = newNodes.pop();
          let extraNodes;
          if (node.text.length === 1) {
            extraNodes = [];
          } else {
            node.text = node.text.substring(0, node.text.length-1);
            node.type = inferType(newNodes, node);
            extraNodes = [node];
          }
          return {
            ...state,
            nodes: newNodes.concat([...extraNodes, { type: "cursor", id: -1}])
          }
        }
        default: {
          const newNodes = [...state.nodes];
          newNodes.pop(); // get rid of cursor in a dumb way need to actually care about location
          if (newNodes.length === 0) {
            newNodes.push({type: "unknown", text: "", id: state.id})
          }
          let node = newNodes.pop();
          if (node.type === "whitespace" || node.type === "newline") {
            newNodes.push(node);
            node = {type: "unknown", text: "", id: state.id};
          }
          node.text += action.key
          // Need to think about context here and if the node should
          // really be inserted in an unknown state.
          // Cursor management is going to be really important going forward.
          node.type = inferType(newNodes, node);
          return {
            ...state,
            id: state.id + 1,
            nodes: newNodes
              .concat([node, { type: "cursor", id: -1}])
          }
        }
      }
    }
  }
  return state
}

// TODO
// Make cursor absolute positioned or rendered some better way
// I mean cursors need to be not nodes themselves, but inside a node.
// So this system makes no sense.


// Up Next:
// Actually handle cursor position
// Decide on a language I am supporting
// Show node types and make things configurable
// Support errors
// Actually have a tree right than a flat list
// Should automatically close brackets.
// Support functions
// Make a pseudo-mode for node selection
// Think about invisible things like , and how to deal with that and space
// Think about selection
// Think about layout



const Home = () => {
  useKeyPress(e => dispatch({type: "press", key: e.key, event: e}));
  const [state, dispatch] = useReducer(reducer, {
    id: 15,
    nodes: [
      { type: "let", text: "let", id: 0 },
      { type: "whitespace", text: " ", id: 1 },
      { type: "identifier", text: "x", id: 2 },
      { type: "whitespace", text: " ", id: 3 },
      { type: "identifier", text: "=", id: 4 },
      { type: "whitespace", text: " ", id: 5 },
      { type: "integer", text: "2", id: 6 },
      { type: "newline", text: "\n", id: 7 },
      { type: "let", text: "let", id: 8 },
      { type: "whitespace", text: " ", id: 9 },
      { type: "identifier", text: "y", id: 10 },
      { type: "whitespace", text: " ", id: 11 },
      { type: "identifier", text: "=", id: 12 },
      { type: "whitespace", text: " ", id: 13 },
      { type: "integer", text: "3", id: 14 },
      { type: "cursor", id: -1},
      
    ],
  });
  return (
    <pre>
      {state.nodes.map(({text, id, type}) => <Node text={text} id={id} type={type} key={id} />)}
    </pre>
  )
}

export default Home;