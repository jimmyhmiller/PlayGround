import React, { useState, useEffect } from 'react'
import Link from 'next/link'
import Head from 'next/head'


const useMouseMove = (drag) => {
  const [position, setPosition] = useState({});
  useEffect(() => {
    const onMouseMove = (e) => setPosition({x: e.clientX, y: e.clientY})
    if (drag) {
      document.addEventListener('mousemove', onMouseMove);
    }
    () => {
      setPosition({})
      document.addEventListener('mousemove', onMouseMove)
    }
  }, [drag])
  return position;
}

const figureOutYOffset = ({ offset, dragging, id }) => {
  const yOffset = 60;
  if (dragging === undefined) {
    return 0;
  }
  if (offset === 0) {
    return 0;
  } else if (offset < 0) {
    return id < dragging && id >= (dragging + offset) ? yOffset : 0
  }
  else if (offset > 0) {
    return id > dragging && id <= (dragging + offset) ? -yOffset : 0
  }
}

const determineTranslation = ({ dragging, id, mousePosition, startPoint }) => {
  if (dragging === id) {
    return `translate(${mousePosition.x  - startPoint.x || 0}px, ${mousePosition.y - startPoint.y || 0}px`
  }
  const diff = mousePosition.y - startPoint.y;
  const offset = Math.round(diff / 60);
  const actualYOffset = figureOutYOffset({ offset, dragging, id })
  return `translate(0px,${actualYOffset}px)`;
}

const useDrag = () => {
  const [dragging, setDragging] = useState(undefined);
  const [startPoint, setStartPoint] = useState({});
  const mousePosition = useMouseMove(dragging);
  return (id) => ({
    props: {
      onMouseDown: (e) => {
        setDragging(id)
        setStartPoint({x: e.clientX, y: e.clientY})
      },
      onMouseUp: (e) => {
        setDragging(undefined)
        setStartPoint({});
      },
      style: { 
        userSelect: "none",
        transition: id === dragging ? "none" : "transform 0.2s cubic-bezier(0.2, 0, 0, 1)",
        transform: determineTranslation({ dragging, id, mousePosition, startPoint })
      },
    },
    dragging,
    startPoint,
    mousePosition,
  })
}

const Item = ({ children, style, ...props }) => (
  <div
    style={{
      display: "flex",
      justifyContent: "center",
      alignItems: "center",
      width: 100,
      height: 50,
      textAlign: "center",
      borderRadius: 2,
      margin: 10,
      backgroundColor: "white",
      ...style,
    }}
    {...props}
  >
    {children}
  </div>
);

const Home = () => {
  const drag = useDrag();
  return (
    <div>
      <Head>
        <title>Home</title>
      </Head>
      <div style={{backgroundColor: "rgb(235, 236, 240)", paddingTop: 5, paddingBottom: 5, width: 120}}>
        <Item {...drag(1).props}>Item 1</Item>
        <Item {...drag(2).props}>Item 2</Item>
        <Item {...drag(3).props}>Item 3</Item>
        <Item {...drag(4).props}>Item 4</Item>
      </div>

    </div>
  )
}

export default Home
