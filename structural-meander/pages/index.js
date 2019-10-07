import React, { useState, useRef, useEffect } from 'react'
import Head from 'next/head'
import AutosizeInput from 'react-input-autosize';
import { setIn, push } from 'zaphod/compat';


const Cell = React.forwardRef(({ text, setText, onKeyDown}, ref) => 
  <AutosizeInput
    ref={ref}
    value={text}
    onKeyDown={onKeyDown}
    onChange={(e) => setText(e.target.value)}
    style={{backgroundColor: "#eee"}} />
)

const Vector = () => {
  const [cells, setCells] = useState([{text: ""}])
  const refs = useRef(new Map());
  const [focusCell, setFocusCell] = useState(null);
  useEffect(() => {
    if (focusCell !== null && refs.current.get(focusCell)) {
      refs.current.get(focusCell).focus()
      setFocusCell(null);
    }
  }, [refs.current, focusCell])

  return (
    <div style={{backgroundColor: "#ccc", padding: 20, fontSize: 20}}>
      [
        {cells.map((cell, i) => (
          <Cell
            key={i}
            ref={inst => inst === null ? refs.current.delete(i) : refs.current.set(i, inst)}
            {...cell}
            onKeyDown={(e) => {
              // What if we want to add something to the beginning?
              if (e.keyCode === 37 && e.target.selectionStart === 0 && i !== 0) {
                setFocusCell(i-1);
              }

              if (e.keyCode === 32 || e.keyCode === 9 || (e.keyCode === 39 && e.target.selectionStart === e.target.value.length)) {
                e.preventDefault();
                if (!cells[i+1]) {
                  setCells(cells => push(cells, {text: ""}));
                }
                setFocusCell(i+1);
              }
            }}
            setText={(text) => setCells(cells => setIn(cells, [i, "text"], text))} />
        ))}
      ]
    </div>
   )
}

const Home = () => (
  <Vector />
)

export default Home
