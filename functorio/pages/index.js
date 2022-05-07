import Head from "next/head";
import Image from "next/image";
import styles from "../styles/Home.module.css";
import { useState, useEffect, useCallback } from "react";

const gridSize = 50;

const Building = ({ x, y }) => {
  return (
    <rect
      stroke="#000"
      fill="#fff"
      x={x * gridSize}
      y={y * gridSize}
      width={gridSize}
      height={gridSize}
    />
  );
};

const HorizontalConnector = ({ x, y}) => {
  return (
    <>
      <rect x={x * gridSize} y={y * gridSize + 6} width={gridSize} height={1} />
      <rect x={x * gridSize} y={(y + 1) * gridSize - 6} width={gridSize} height={1} />
    </>
  )
}

const VerticalConnector = ({ x, y}) => {
  return (
    <>
      <rect x={x * gridSize + 6} y={y * gridSize} width={1} height={gridSize} />
      <rect x={(x + 1) * gridSize - 6}  y={y * gridSize} width={1} height={gridSize} />
    </>
  )
}


const RightDown = ({ x, y}) => {
  return (
    <>
      <rect x={x * gridSize} y={y * gridSize + 6} width={gridSize - 6} height={1} />
      <rect x={x * gridSize} y={(y + 1) * gridSize - 6} width={6} height={1} />
      <rect x={x * gridSize + 6} y={(y + 1) * gridSize - 6} width={1} height={6} />
      <rect x={(x + 1) * gridSize - 6}  y={y * gridSize + 6} width={1} height={gridSize - 6} />
    </>
  )
}
const UpRight = ({ x, y}) => {
  return (
    <>
      <rect x={x * gridSize + 6} y={y * gridSize + 6} width={1} height={gridSize - 6} />
      <rect x={x * gridSize + 6} y={y * gridSize + 6} width={gridSize - 6} height={1} />
      <rect x={(x + 1) * gridSize - 6}  y={(y + 1) * gridSize - 6} width={1} height={6} />
      <rect x={(x + 1) * gridSize - 6} y={(y + 1) * gridSize - 6} width={6} height={1} />
    </>
  )
}

const DownRight = ({ x, y}) => {
  return (
    <>
      <rect x={x * gridSize + 6} y={y * gridSize} width={1} height={gridSize-6} />
      <rect x={x * gridSize + 6} y={(y + 1) * gridSize - 6} width={gridSize-6} height={1} />
      <rect x={(x + 1) * gridSize - 6} y={y * gridSize} width={1} height={6} />
      <rect x={(x + 1) * gridSize - 6} y={y * gridSize + 6} width={6} height={1} />
    </>
  )
}

// Why in the world is this different from the others?
const RightUp = ({ x, y}) => {
  return (
    <>
      <rect x={x * gridSize} y={(y + 1) * gridSize - 6} width={gridSize - 6} height={1} />
      <rect x={(x + 1) * gridSize - 6}  y={y * gridSize} width={1} height={gridSize - 5} />
     
      <rect x={x * gridSize} y={y * gridSize + 6} width={7} height={1} />
      <rect x={x * gridSize + 6} y={y * gridSize} width={1} height={6} />
    </>
  )
}

// Why in the world is this different from the others?
const DownLeft = ({ x, y}) => {
  return (
    <>
      <rect x={(x + 1) * gridSize - 6}  y={y * gridSize} width={1} height={gridSize - 6} />
      <rect x={x * gridSize} y={(y + 1) * gridSize - 6} width={gridSize - 5} height={1} />
      <rect x={x * gridSize + 6} y={y * gridSize} width={1} height={6} />
      <rect x={x * gridSize} y={y * gridSize + 6} width={7} height={1} />
    </>
  )
}

const randElem = (items) => items[Math.floor(Math.random()*items.length)]

const elems = [Building, HorizontalConnector, RightDown, VerticalConnector, DownRight, UpRight, DownLeft]



const CompletelyRandom = () => {
  let allElems = [];
  for (let x = 0; x < 100; x++) {
    for (let y = 0; y < 100; y++) {
      const Elem = randElem(elems);
      allElems.push(<Elem key={`x:${x},y${y}`} x={x} y={y} />)
    }
  }
  return allElems;
}

const DrawBoard = ({board}) => {
  let pieces = [];
  for (let [x, ys] of Object.entries(board)) {
    for (let [y, {Component}] of Object.entries(ys)) {
      pieces.push(<Component key={`${x},${y}`} x={parseInt(x, 10)} y={parseInt(y, 10)} />)
    }
  }
  return pieces
}


// TODO: Need to draw corners
// Do I need some notion of history? Or should it be determined by the area around?
// Not really sure


export default function Home() {

  const [board, setBoard] = useState({});
  const [isMouseDown, setMouseDown] = useState(false);

  const onMouseDown = useCallback((e) => {
    setMouseDown(true);
    const x = Math.floor(e.clientX / gridSize);
    const y = Math.floor(e.clientY / gridSize);

    setBoard(board => {
      return {
        ...board,
        [x]: {
          ...(board[x] || {}),
          [y]: {Component: Building}
        }
      }
    });
  }, [])

  const onMouseUp = useCallback((e) => {
    setMouseDown(false);
  }, [])

  const onMouseMove = useCallback((e) => {
    const x = Math.floor(e.clientX / gridSize);
    const y = Math.floor(e.clientY / gridSize);

    if (isMouseDown) {

      setBoard(board => {
        if ((Math.abs(e.movementX) + Math.abs(e.movementY) < 3) || (board[x] && board[x][y] && board[x][y] && board[x][y].Component === Building)) {
          return board
        }
        return {
          ...board,
          [x]: {
            ...(board[x] || {}),
            [y]: {Component: Math.abs(e.movementX) > Math.abs(e.movementY) ? HorizontalConnector : VerticalConnector}
          }
        }
      });
    }
    // console.log("mouseMove", e.clientX, e.clientY, e.movementX, e.movementY);
  }, [isMouseDown])

  return (
    <svg onMouseDown={onMouseDown} onMouseUp={onMouseUp} onMouseMove={onMouseMove} width={3000} height={3000}>
     <DrawBoard board={board} />
    </svg>
  );
}