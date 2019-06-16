import React, { useState, useEffect } from 'react';
import { hot } from 'react-hot-loader'
import Inspector from 'react-inspector';
import toTime from 'to-time';
import orderBy from 'lodash.orderby';
import groupBy from 'lodash.groupby';
import map from 'lodash.map';
import range from 'lodash.range';
import random from 'random-seed';

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

const ArticleInfo = ({ is_article, word_count }) => {
  if (is_article === "1" && word_count) {
    return (
      <Small>{word_count} words</Small>
    )
  }
  return null
}

function nWords(str, n) {
  return str.split(/\s+/).slice(0, n).join(" ");
}

const Conditional = ({ exists, children }) => {
  if (exists) {
    return children
  }
  return null;
}

const ItemTitle = inspect(
  ({resolved_title, given_title, excerpt, resolved_url, onClick }) =>
    <span onClick={onClick} style={{maxWidth: 800, overflow: "hidden", textOverflow: "ellipsis"}}>
      {resolved_title || given_title || nWords(excerpt, 5) || resolved_url}
    </span>
)

const siteName = ({ resolved_url, given_url, domain_metadata }) =>
  (domain_metadata && domain_metadata.name) ||
  (resolved_url && new URL(resolved_url).hostname) ||
  (given_url && new URL(given_url).hostname) ||
  "WHAT?"

const Item = ({ onClick, tags, excerpt, given_url, resolved_url, domain_metadata, ...props, }) =>
    <>
      <li>
        <ItemTitle excerpt={excerpt} {...props} resolved_url={resolved_url} />
        {" "}(<a href={resolved_url}>{siteName({ resolved_url, domain_metadata, given_url })} </a>)

        <ArticleInfo {...props} />

        <Conditional exists={tags}>
          <div style={{padding:0, marginTop:-5, fontSize: 11, color: "gray"}}>
            tags: {tags && Object.keys(tags).join(", ")}
          </div>
        </Conditional>

        <Conditional exists={excerpt}>
          <div style={{width: 600, paddingTop: 10, paddingLeft: 10, fontSize: 13, color: "gray"}}>
            {excerpt}
          </div>
        </Conditional>
      </li>
    </>

const totalWords = (items) =>
  items.reduce((total, item) => total + parseInt(item.word_count || 0, 10), 0)
  .toLocaleString()

const totalTime = (items) => {
  const minutes = items.reduce((total, item) => total + (item.time_to_read || 0), 0);
  return toTime.fromMinutes(minutes).humanize()
}



const AllItems = ({ items }) => {
  const orderedItems = orderBy(items, x => x.tags && Object.keys(x.tags)[0]);

  return (
    <>
      <h1 style={{marginTop: 0}}>Pocket App</h1>
      <Small>Total Words: {totalWords(orderedItems)}</Small>
      <Small>•</Small>
      <Small>Time to Read: {totalTime(orderedItems)}</Small>
      <ul>
        {orderedItems.map(item => <Item key={item.item_id} {...item} />)}
      </ul>
    </>
  )
}

const View = ({ name, selectedView, children }) => {
  return (
    <div style={{display: name === selectedView ? "block" : "none" }}>
      {children}
    </div>
  )
}

const Selector = ({ onClick, text }) => {
  return (
    <a style={{paddingRight: 15}}
       href="#" 
       onClick={(e) => { e.preventDefault(); onClick()} }
     >{text}</a>
  )
}



// Make it so I can navigate between days.
// I also need to show other things from the site.
// I need to show 3? selections each with different reading times.

const Experiment1 = ({ items }) => {
  const filteredItems = items.filter(item => item.is_article === "1")
  const groupedItems = groupBy(filteredItems, item => siteName(item))
  const pairGroup = map(groupedItems, (value, key) => [key, value])
  const orderedItems = map(orderBy(pairGroup, x => x[1].length, "desc"), x => x[1][0])
  const gen = random(new Date().toLocaleDateString())
  const pickItem = (coll) => coll[gen.intBetween(0, coll.length-1)]
  const randomItems = range(0, Math.min(pairGroup.length,10)).map(x => pickItem(pickItem(pairGroup)[1]))
  return (
    <>
      <h1 style={{marginTop: 0}}>Pocket App</h1>
      <ul>
        {randomItems.map(item => <Item key={item.item_id} {...item} />)}
      </ul>
    </>
  )
}


const App = () => {

  const items = orderBy(
    useLocalCache([], "/api/items?count=10000", x => Object.values(x.list)),
  )

  const [selectedView, setView] = useState("experiment1");

  return (
    <>
      <div>
        <Selector onClick={() => setView("all")} text="All" />
        <Selector onClick={() => setView("experiment1")} text="Experiment 1" />
      </div>
      <View name="all" selectedView={selectedView}>
        <AllItems items={items} />
      </View>
      <View name="experiment1" selectedView={selectedView}>
        <Experiment1 items={items} />
      </View>
    </>
  ) 
}


export default hot(module)(App);
