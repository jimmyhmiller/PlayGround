import Head from 'next/head'
import { useState, useReducer, useEffect, useMemo } from 'react';

// This is in the write ugly code to get something working phase.


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

const hasPadding = ({ type }) => 
 type !== "whitespace" && type !== "newline" && type !== "cursor";

const Node = ({ text, type }) => {
  const hasBorder = type !== "whitespace" && type !== "newline" && type !== "cursor";
  const color = colors[type] || defaultColor;
  return (
    <span 
      style={{
        color: color,
        padding: hasPadding({ type }) ? 3 : undefined,
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
  } else if (currentNode.text.match(/^[A-Za-z=\-+&|\/\?][A-Za-z&|\/\?0-9]*$/)) {
    return "identifier"
  }
  else {
    return "unknown"
  }
}


const splitAtCursor = ({nodes, cursor}) => {
  const newNodes = [...nodes];
  return [newNodes.slice(0, cursor[0] + 1), newNodes.slice(cursor[0]+1)]
}

const atEndOfNodeRight = ({nodes, cursor}) => {
  const [cursorNode, cursorIndex] = cursor;
  const node = nodes[cursorNode];
  return node.text.length < cursorIndex + 1
}

const atEndOfNodeLeft = ({nodes, cursor}) => {
  const [cursorNode, cursorIndex] = cursor;
  const node = nodes[cursorNode];
  return cursorIndex - 1 < 0;
}


const reducer = (state, { type, ...action }) => {

  switch (type) {
    case "press": {
      switch (action.key) {
        case "ArrowRight": {
          const [cursorNode, cursorIndex] = state.cursor;
          if (cursorNode + 1 >= state.nodes.length) {
            break;
          }
          const node = state.nodes[cursorNode];
          const atNodeEnd = node.text.length < cursorIndex + 1;
          // console.log(atNodeEnd, cursorNode, cursorIndex, node.text.length)
          return {
            ...state,
            // need to handle moving new lines
            // need to handle moving nodes
            cursor: [
              atNodeEnd ? cursorNode + 1 : cursorNode,
              atNodeEnd ? 1 : cursorIndex + 1
            ]
          }
        }
        case "ArrowLeft": {
          const nodes = state.nodes;
          const [cursorNode, cursorIndex] = state.cursor;
          const node = state.nodes[cursorNode];
          const atNodeEnd = cursorNode === 0 ? cursorIndex <= 0 : cursorIndex - 1 <= 0;
          if (atNodeEnd && cursorNode - 1 < 0) {
            break;
          }
          // console.log(atNodeEnd, cursorNode, cursorIndex, node.text.length)
          return {
            ...state,
            cursor: [
              atNodeEnd ? cursorNode - 1 : cursorNode,
              atNodeEnd ? nodes[cursorNode - 1].text.length : cursorIndex - 1
            ]
          }
        }

        // Need to do up and down.

        case "Shift":
        case "Meta":
        case "Control":
        case "Alt":
        case "Escape": {
          return state
        }
        case " ":
        case "Enter": {
          const [left, right] = splitAtCursor(state);
          const rightEnd = atEndOfNodeRight(state);
          const leftEnd = atEndOfNodeLeft(state);
          const newNode = { 
            type: action.key === " " ? "whitespace" : "newline",
            text: action.key === " " ? " " : "\n",
            id: state.id,
          }
          // maybe don't want this behavior?
          if (rightEnd && right[0] && right[0].type === action.key) {
            return {
              ...state,
              cursor: [state.cursor[0]+1, 1]
            }
          } else if (rightEnd) {
            return {
              ...state,
               id: state.id + 1,
              cursor: [state.cursor[0]+1, 1],
              nodes: left.concat([newNode, ...right])
            }
          } else if (leftEnd) {
            return {
              ...state,
               id: state.id + 1,
              cursor: [state.cursor[0], 1],
              nodes: [newNode].concat([...left, ...right])
            }
          } else {
            const node = left.pop();
            const leftNode = {...node}
            const rightNode = {...node};
            leftNode.text = node.text.substring(0, state.cursor[1])
            leftNode.type = inferType(state.nodes, leftNode);
            rightNode.text = node.text.substring(state.cursor[1]);
            rightNode.type = inferType(state.nodes, rightNode);
            rightNode.id = state.id;
            return {
              ...state,
               id: state.id + 2,
              cursor: [state.cursor[0]+1, 1],
              nodes: left.concat([leftNode, { ...newNode, id: state.id +1 }, rightNode, ...right])
            }
          }
        }
        // backspace isn't quite right right
        case "Backspace": {
          action.event.preventDefault();
          if (state.nodes.length === 0) {
            break;
          }
          const [left, right] = splitAtCursor(state);
          const node = left.pop();
          let extraNodes;
          if (node.text.length === 1) {
            extraNodes = [];
          } else {
            const leftText = node.text.substring(0, state.cursor[1]-1)
            const rightText = node.text.substring(state.cursor[1]);
            node.text = `${leftText}${rightText}`
            node.type = inferType(state.nodes, node);
            extraNodes = [node];
          }

          const isLastNode = state.cursor[0] - 1 < 0
          const characterIndex = isLastNode ? 0 : state.nodes[state.cursor[0] - 1].text.length
          return {
            ...state,
            cursor: [
              extraNodes.length >= 1 ? state.cursor[0] : Math.max(state.cursor[0] - 1, 0),
              extraNodes.length >= 1 ? Math.max(state.cursor[1] - 1, 0) : characterIndex
            ],
            nodes: left.concat([...extraNodes, ...right])
          }
        }
        default: {
          const [left, right] = splitAtCursor(state);
          if (left.length === 0 && right.length === 0) {
            left.push({type: "unknown", text: "", id: state.id})
          }
          let node = left.pop();
          let addedNewNode = false;
          if (node.type === "whitespace" || node.type === "newline") {
            left.push(node);
            addedNewNode = true;
            node = {type: "unknown", text: "", id: state.id};
          }
          const leftText = node.text.substring(0, state.cursor[1])
          const rightText = node.text.substring(state.cursor[1]);
          node.text = `${leftText}${action.key}${rightText}`
          // Need to think about context here and if the node should
          // really be inserted in an unknown state.
          // Cursor management is going to be really important going forward.
          node.type = inferType(state.nodes, node);
          return {
            ...state,
            cursor: [
               addedNewNode ? state.cursor[0] + 1 : state.cursor[0],
               addedNewNode ? 1 : state.cursor[1] + 1
            ], 
            id: state.id + 1,
            nodes: left.concat([node, ...right])
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
// Need to make nodes for reasons other than space :)
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

// Need to have nice debugging of state of the editor


// Need better bounds checking!




// Needs lots of clean up
const calculateCursorPosition = ({ nodes, cursor: [cursorNode, cursorIndex], charWidth, nodePaddingX,  charHeight, offsetX, offsetY, nodePaddingY }) => {
  if (nodes.length <= 0) {
    return [offsetX, offsetY]
  }
  const node = nodes[cursorNode];
  const currentIsNewline = node.type === "newline";
  const halfPadding = nodePaddingX/2
  const padding = !currentIsNewline && hasPadding(node) ? halfPadding : 0;
  let x = 0
  let y = 0
  for (let i = 0; i < cursorNode; i++) {
    if (nodes[i].type === "newline") {
      y += charHeight + nodePaddingY
      x = 0;
    } else if (!currentIsNewline) {
      x += (nodes[i].text.length * charWidth) + (hasPadding(nodes[i]) ? nodePaddingX : 0)
    }
  }
  x += currentIsNewline ? offsetX : offsetX + padding + (cursorIndex * charWidth);
  y += offsetY + (currentIsNewline ? charHeight + nodePaddingY : 0);
  return [x, y]
}

const Home = () => {
  useKeyPress(e => dispatch({type: "press", key: e.key, event: e}));
  const [offsetX, offsetY] = [8, 24];
  const [charWidth, charHeight] = [12, 20.5]
  const [nodePaddingX, nodePaddingY] = [8, 14.5];
  const [state, dispatch] = useReducer(reducer, {
    id: 15,
    cursor: [0, 0],
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
    ],
  });
  const [cursorX, cursorY] = useMemo(
    () =>
      calculateCursorPosition({
        nodes: state.nodes,
        cursor: state.cursor,
        charWidth,
        charHeight,
        offsetX,
        offsetY,
        nodePaddingX,
        nodePaddingY,
      }),
    [state.nodes, state.cursor, offsetX, offsetY, charWidth, charHeight, nodePaddingX, nodePaddingY]
  );
  return (
    <>
      <span className="cursor" style={{position: "absolute", top: cursorY, left: cursorX, height: charHeight}} />
      <pre>
        {state.nodes.map(({text, id, type}) => <Node text={text} id={id} type={type} key={id} />)}
      </pre>
    </>
  )
}

export default Home;