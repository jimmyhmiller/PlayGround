import React, { useState, useEffect } from 'react';
import Inspector from 'react-inspector';
import toTime from 'to-time';
import orderBy from 'lodash.orderby';
import groupBy from 'lodash.groupby';
import map from 'lodash.map';
import range from 'lodash.range';
import random from 'random-seed';
import Head from 'next/head'
import * as localForage from 'localforage';
 
const useLocalStorage = (key) => {
  const [value, setValue] = useState([]);
  const [offset, setOffset] = useState( 0);

  useEffect(() => {
    (async () => {
        const initialValue = await localForage.getItem(key);
        const initialOffset = await localForage.getItem("offset");
        setValue(initialValue)
        setOffset(initialOffset)
    })()
  }, [])

  useEffect(() => {
    if (key && value) {
      localForage.setItem(key, value);
    }
  }, [key, value])


  useEffect(() => {
    if (key && value) {
      localForage.setItem("offset", offset);
    }
  }, [offset])

  return {value, setValue, offset, setOffset}
}


const useFetchPaginate = (initial, endpoint, count, initialOffset, setLoading, f) => {
  const [data, setData] = useState(initial || []);
  const [offset, setOffset] = useState(initialOffset);
  const [complete, setComplete] = useState(false);

  useEffect(() => {
    if (complete) {
      setLoading(null);
    }
    if (endpoint && !complete) {
      setLoading(`Loading: ${offset}`);
      fetch(`${endpoint}?count=${count}&offset=${offset}`, {
          credentials: "same-origin"
        })
        .then(resp => {
          if (resp.status === 401 && !window.location.toString().includes("oauth")) {
            if (window.location.toString().includes("localhost")) {
              console.log("Set the Auth Token!", process.env.VERCEL_ENV);
              return
            }
            window.location = "/api/oauth"
          }
          else if (resp.status !== 200) {
            return initial
          }
          return resp.json()
        })
        .then(f)
        .then(newData => {
          if (newData.length === count) {
            setData(data.concat(newData));
            setOffset(offset + count);
          } else {
            setData(data.concat(newData));
            setOffset(offset + newData.length);
            setComplete(true);
          }
        })
        .catch(e => {
          console.error(e);
          setLoading("Error");
        })
      }
  }, [endpoint, offset, complete])

  return complete ? [data, offset] : [];
}


const useLocalCache = (initial, endpoint, setLoading, f) => {
  const {value, setValue, offset, setOffset} = useLocalStorage(endpoint);
  // ugly hard coding of endpoint to fetch
  const [data, newOffset] = useFetchPaginate(value, endpoint, 1000, offset, setLoading, f);
  useEffect(() => {
    if (data) {
      setValue(data)
      setOffset(newOffset);
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

const Small = ({ children, style={}, ...props }) =>
  <small {...props} style={{fontSize: 11, color: "gray", paddingLeft: 5, paddingRight: 5, ...style}}>
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
  return str?.split(/\s+/).slice(0, n).join(" ");
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

const sendAction = ({ action, item_id }) => {
  return fetch(`/api/actions?action=${action}&item_id=${item_id}`, {
    credentials: "same-origin"
  })
}

const Item = ({ onClick, tags, excerpt, given_url, resolved_url, domain_metadata, ...props }) =>
  <>

    <li>
      <ItemTitle excerpt={excerpt} {...props} resolved_url={resolved_url} />
      {" "}(<a href={resolved_url}>{siteName({ resolved_url, domain_metadata, given_url })} </a>)

      <ArticleInfo {...props} />
      <Small
        className="hover">
        <a style={{cursor:"pointer"}} onClick={(e) => { e.preventDefault(); sendAction({item_id: props.item_id, action: "archive"})}}>
      ✓
      </a>
      </Small>

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
  const randomItems = range(0, Math.min(pairGroup.length, 10)).map(x => pickItem(pickItem(pairGroup)[1]))
  return (
    <>
      <h1 style={{marginTop: 0}}>Pocket App</h1>
      <ul>
        {randomItems.map(item => <Item key={item.item_id} {...item} />)}
      </ul>
    </>
  )
}


const Experiment2 = ({ items }) => {
  const filteredItems = orderBy(items.filter(item => item.tags && item.tags.muse), 'time_added', 'desc');
  return (
    <>
      <h1 style={{marginTop: 0}}>Pocket App</h1>
      <ul>
        {filteredItems.map(item => <Item key={item.item_id} {...item} />)}
      </ul>
    </>
  )
}


const App = () => {

  const [loading, setLoading] = useState("not loading");
  const items = orderBy(
    useLocalCache([], "/api/items", setLoading, x => x ? Object.values(x.list) : []),
  )


  const [selectedView, setView] = useState("experiment1");

  return (
    <>
      <Head>
        <title>Pocket App</title>
        <link href="https://unpkg.com/superstylin@1.0.3/src/index.css" rel="stylesheet" />
      </Head>
       <style jsx global>{`
          .hover {
            opacity: 0;
          }
          .hover:hover {
            opacity: 1;
          }
        `}
      </style>
      <div style={{position:"absolute", bottom: 100, right: 100}}>{loading}</div>
      <div>
        <Selector onClick={() => setView("all")} text="All" />
        <Selector onClick={() => setView("experiment1")} text="Experiment 1" />
        <Selector onClick={() => setView("muse")} text="Muse" />
      </div>
      <View name="all" selectedView={selectedView}>
        <AllItems items={items} />
      </View>
      <View name="experiment1" selectedView={selectedView}>
        <Experiment1 items={items} />
      </View>
      <View name="muse" selectedView={selectedView}>
        <Experiment2 items={items} />
      </View>
    </>
  ) 
}


export default App;
