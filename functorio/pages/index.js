import Head from "next/head";
import Image from "next/image";
import styles from "../styles/Home.module.css";

const gridSize = 30;

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


export default function Home() {
  return (
    <svg width={3000} height={3000}>
      <CompletelyRandom />
    </svg>
  );
}