import Head from 'next/head'
import { useState, useMemo } from 'react';
import {useTransition, animated} from 'react-spring'
import { sortBy } from 'lodash';

const Box = ({ i }) => 
  <div 
    style={{
      border: "1px solid black",
      height: 200,
      margin: 10
    }}>
    {i}
  </div>

const ThreeBox = animated(({ style, i, ...props }) => {
  const [toggle, setToggle] = useState(false);
  const styles = {...style};
  return (
    <div onClick={() => setToggle(!toggle)} style={{display: "grid", gridTemplateColumns: "1fr 1fr 1fr", ...styles}} {...props}>
      <Box i={i} />
      <Box i={i} />
      <Box i={i} />
    </div>
  )
})

// Need to restore scroll when unclicking.

export default function Home() {
  const [items, setItems] = useState([...Array(5).keys()]);
  const transitions = useTransition(items, item => item, {
    from: { opacity: 0, y: 0},
    enter: (item) => ({ opacity: 1, y: 0}),
    leave: { opacity: 0 },
    update: (item) =>({y: items.length !== 1 ? 0 : item*-222 + window.scrollY}),
    unique: true
  })
  const orderedTransitions = useMemo(() => {
    return sortBy(transitions, 'item');
  }, [transitions])
  return (
    <>
      <Head>
        <title>Create Next App</title>
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main>
        {orderedTransitions.map(({ item, key, props: {y, ...props} }) =>
          <ThreeBox onClick={() => {
            if (items.length > 1) {
              setItems([item])
            } else {
              setItems([...Array(5).keys()])
            }
        }} i={item} key={item} style={{...props, position: "relative", top: orderedTransitions.length === 1 ? 0 : y.interpolate(y => `${y}px`)}} />
        )}
      </main>
    </>
  )
}
