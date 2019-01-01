import React, { useState, useEffect } from 'react';
import Inspector from 'react-inspector';
import toTime from 'to-time';

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

const useLocalStorage = (key) => {

  const initialValue = localStorage[key] && JSON.parse(localStorage[key])

  const [value, setValue] = useState(initialValue);

  useEffect(() => {
    if (key && value) {
      localStorage[key] = JSON.stringify(value);
    }
  }, [key, value])

  return [value, setValue]
}

const useFetchData = (initial, endpoint, f) => {
  const [data, setData] = useState(initial);

  useEffect(() => {
    if (endpoint) {
      fetch(endpoint, { 
          credentials: "same-origin"
        })
        .then(resp => {
          if (resp.status === 401 && !window.location.toString().includes("oauth")) {
            window.location = "/oauth"
          }
          else if (resp.status !== 200) {
            return initial
          }
          return resp.json()
        })
        .then(f)
        .then(data => setData(data))
      }
  }, [endpoint])

  return data;
}

const useLocalCache = (initial, endpoint, f) => {
  const [value, setValue] = useLocalStorage(endpoint);
  const endpointToFetch = value ? undefined : endpoint;
  const data = useFetchData(undefined, endpointToFetch, f);
  useEffect(() => {
    if (!value && data) {
      setValue(data)
    }
  }, [value, data])
  return value || data || initial
}

const inspect = (comp) => (props) => {
  const [show, setShow] = useState(false);
  const onClick = (e) => {
    if (e.metaKey) {
      setShow(!show)
    }
  }
  return <>
    {comp({onClick, ...props})}
    {show && <Inspector data={props} />}
  </>
}

const Small = ({ children }) => 
  <small style={{fontSize: 11, color: "gray", paddingLeft: 5, paddingRight: 5}}>
    {children}
  </small>

const article = ({ is_article, word_count }) => {
  if (is_article === "1" && word_count) {
    return (
      <Small>{word_count} words</Small>
    )
  }
  return undefined
}

const decorate = (comp) => (props) => {
  const articleDecorate = article(props);
  return comp({decorate: articleDecorate, ...props});
}

function nWords(str, n) {
  return str.split(/\s+/).slice(0, n).join(" ");
}

const Item = decorate(inspect(({ given_title, resolved_title, item_id, resolved_url, decorate, onClick, excerpt }) => 
  <>
    <li onClick={onClick}>
      <span style={{maxWidth: 800, overflow: "hidden", textOverflow: "ellipsis"}}>
        {resolved_title || given_title || nWords(excerpt, 5) || resolved_url}
      </span>
      {" "}(<a href={resolved_url}>link</a>)
      {decorate}
    </li>
  </>
))

const totalWords = (items) => 
  items.reduce((total, item) => total + parseInt(item.word_count || 0, 10), 0)
  .toLocaleString()

const totalTime = (items) => {
  const minutes = items.reduce((total, item) => total + (item.time_to_read || 0), 0);
  return toTime.fromMinutes(minutes).humanize()
}


const App = () => {
  const items = useLocalCache([], "/api/items?count=10000", x => Object.values(x.list))
  return (
    <>
      <h1>Pocket App</h1>
      <Small>Total Words: {totalWords(items)}</Small>
      <Small>•</Small>
      <Small>Time to Read: {totalTime(items)}</Small>
      <ul>
        {items.map(item => <Item key={item.item_id} {...item} />)}
      </ul>
    </>
  )
}

export default App;
