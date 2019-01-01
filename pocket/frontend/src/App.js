import React, { useState, useEffect } from 'react';

// const useDebounce = (value, delay) => {
//   const [currentValue, setCurrentValue] = useState(value);

//   useEffect(() => {
//     const delayed = setTimeout(() => {
//       setCurrentValue(value)
//     }, delay);

//     return () => {
//       clearInterval(delayed);
//     }
//   }, [value, delay])

//   return currentValue;
// } 



const useFetchData = (initial, endpoint, f) => {
  const [data, setData] = useState(initial);

  useEffect(() => {
    fetch(endpoint, { 
        credentials: "same-origin"
      })
      .then(resp => {
        if (resp.status === 401) {
          window.location = "/oauth"
        }
        else if (resp.status !== 200) {
          return initial
        }
        return resp.json()
      })
      .then(f)
      .then(data => setData(data))
  }, [endpoint])

  return data;
}

const Item = ({ given_title, item_id, given_url }) => <li>{given_title || given_url}</li>

const App = () => {
  const items = useFetchData([], "/api/items?count=10000", x => Object.values(x.list))
  return (
    <ul>
      {items.map(item => <Item key={item.item_id} {...item} />)}
    </ul>
  )
}

export default App;
