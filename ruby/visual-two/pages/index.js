import { useState, useEffect, useRef, useMemo, Suspense } from 'react';
import dynamic from 'next/dynamic';
import NoSSR from '@mpth/react-no-ssr';
import localforage from 'localforage';



// TODO: Make svg cache
// TODO: get method by name not index


// console.log(d3)

// const Graphviz = dynamic(() => import('graphviz-react'), { ssr: false });

const Graphviz = ({id, dot, options: {width, height}, onRender, cache, setCache}) => {
  let selector = id.replace(/[^A-Za-z\-0-9]/g, "")
  useEffect(() => {

    const renderSvg = (svg) => {
      const div = document.getElementById(selector);
      if (!div) {
        console.error(`no selector ${selector}`)
        return;
      }
      div.innerHTML = svg;
      div.children[0].setAttribute("width", `${width}px`)
      div.children[0].setAttribute("height", `${height}px`)
      onRender && onRender()
    }

    if (cache && cache[selector]) {
      console.log("cache hit")
      renderSvg(cache[selector])
    }

    window["@hpcc-js/wasm"].graphviz.layout(dot, "svg", "dot").then(svg => {
      renderSvg(svg);
      setCache && setCache(cache => ({
        ...cache,
        [selector]: svg
      }))
      
    });
  //   console.log("making graph");

  //   const renderer = d3.select(`#${selector}`)
  //     .graphviz()
    
  //   renderer.options({
  //     useSharedWorker: true,
  //   })
  //   .zoom(false)
  //   .width(width)
  //   .height(height)
  //   .fit(true)
  //   .dot(dot);

  //   renderer.on("end", () => {
  //     renderer?.destroy();  
  //   }).render();
  }, [dot])


  return <div id={selector} style={{width, height}} />

}



const mod = (m, n) => {
  return ((m % n) + n) % n;
};




const useLocalForage = (key, initialValue) => {
  const [storedValue, setStoredValue] = useState(initialValue);

  // useEffect(() => {
  //   (async () => {
  //     const item = await localforage.getItem(key);
  //     if (item) {
  //       setStoredValue(item);
  //     }
  //   })()
  // }, [])


  const setValue = (value) => {
    console.log("before before", value, key);
    (async () => {
      try {
        console.log("before", value)
        const newValue = await localforage.setItem(key, value);
        console.log("value", newValue)
        if (newValue) {
          setStoredValue(newValue);
        }
      } catch (e) {
        console.error("failed to set value");
      }
    })();
  }

  return [storedValue, setValue]

}



// Hook
function useLocalStorage(key, initialValue) {
  // State to store our value
  // Pass initial state function to useState so logic is only executed once
  const [storedValue, setStoredValue] = useState(() => {
    if (typeof window === "undefined") {
      return initialValue;
    }

    try {
      // Get from local storage by key
      const item = window.localStorage.getItem(key);
      // Parse stored json or if none return initialValue
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      // If error also return initialValue
      console.log(error);
      return initialValue;
    }
  });

  // Return a wrapped version of useState's setter function that ...
  // ... persists the new value to localStorage.
  const setValue = (value) => {
    try {
      // Allow value to be a function so we have same API as useState
      const valueToStore =
        value instanceof Function ? value(storedValue) : value;
      // Save state
      setStoredValue(valueToStore);
      // Save to local storage
      if (typeof window !== "undefined") {
        window.localStorage.setItem(key, JSON.stringify(valueToStore));
      }
    } catch (error) {
      // A more advanced implementation would handle the error case
      console.log(error);
    }
  };

  return [storedValue, setValue];
}


const range = (size, startAt=0) => {
  return [...Array(size).keys()].map(i => i + startAt);
}




const prevent = (f) => e => {
  e.preventDefault();
  f(e)
}

const establishWebSocket = ({ onMessage, onCreate }) => {
  console.log("Establishing connection")
  const ws = new WebSocket("ws://127.0.0.1:2794");

  // setInterval(() => {
  //   console.log(ws.readyState)
  // }, 1000)
  ws.addEventListener("open", function(event) {
    console.log("opened")
    ws.send("Hello");
  })
  ws.addEventListener("message",function(event) {
    onMessage(event.data);
  })
  ws.addEventListener("close", function(event) {
    console.log("close");
    establishWebSocket({ onMessage, onCreate });
  })
  ws.addEventListener("error", function(event) {
    console.log("error");
  })
  onCreate(ws);
}


const getInstructionText = (instructions) => {
  return (
    `${instructions.map(({type, value}) => {
      if (type === "comment") {
        return `#${value}`
      } else {
        return value
      }
    }).join(`\\l`)}\\l
  `
)
}

const addEpochEdges = ({edge_list, epoch}) => {
  return edge_list.map(([x, y]) => [`${x}-${epoch}`, `${y}-${epoch}`])
}


const makeGraphs = (blocks) => {

  if (!blocks) {
    return ""
  }
  return `

    ${blocks.map(({hash, instructions, epoch}) => {
      return `"${hash}-${epoch}" [label="${getInstructionText(instructions)}", shape="square"]`
    }).join("\n")}

   
    ${blocks.flatMap(addEpochEdges).map(([x, y]) => `"${x}" -> "${y}"`).join('\n')}

  `
}

const drop = (n, array) => {
  if (!array) {
    return []
  }
  let arr = []
  for (let i = n; i < array.length; i++) {
    arr.push(array[i])
  }
  return arr
}

const take = (n, array) => {
  if (!array) {
    return []
  }
  let arr = [];
  for (let i = 0; i < n && i < array.length; i++) {
    arr.push(array[i])
  }
  return arr
}


const groupBy = (array, f) =>  {
  let results = {}
  for (let x of array) {
    let key = f(x);
    if (results[key]) {
      results[key].push(x);
    } else {
      results[key] = [x]
    }
  }
  return results
}


const makeDot = (blocks) => {
  return `
    digraph CFG {
      graph [splines=ortho, nodesep=2, ranksep=1]
      pad=1
      bgcolor="#ffffff"

      ${makeGraphs(blocks)}

    }

  `
}


const useWebSocket = () => {
  const wsRef = useRef(null);
  const [data, setData] = useState([]);

  useEffect(() => {
    if (!wsRef.current) {
      establishWebSocket({
        onCreate: ws => wsRef.current = ws,
        onMessage: message => setData(data => data.concat(JSON.parse(message).blocks))
      });
    }
    return () => {
      console.log("remount")
      if (wsRef.current) {
        wsRef.current.onclose = () => {};
        wsRef.current.close();
      }
    }
  }, [])
    
  return data;

}


const makeSimplifiedGraph = (blocks) => {
  return `
    digraph CFG {
      graph [splines=ortho, nodesep=2, ranksep=1]
      pad=1
      bgcolor="#ffffff"

      ${blocks.map(({hash, instructions, epoch}) => {
        return `"${hash}-${epoch}" [label="block", shape="square"]`
      }).join("\n")}

      ${blocks.flatMap(addEpochEdges).map(x => x[1]).filter(x => x.length < 18).map(x => `"${x}" [label="stub"]`).join("\n")}

      ${blocks.flatMap(addEpochEdges).map(([x, y]) => `"${x}" -> "${y}"`).join('\n')}

    }

  `
}

const sortBy = (array, f) => {
  const temp = [...array];
  temp.sort((a, b) => f(a) - f(b))
  return temp;
}

const partitionBy = (array, f) => {
  if (!array || array.length === 0) {
    return []
  }
  let first = array.shift();
  let results = [[first]];
  let lastValue = f(first);
  let index = 0;

  for (let elem of array) {
    let value = f(elem)
    if (value === lastValue) {
      results[index].push(elem)
    } else {
      lastValue = value
      results.push([elem]);
      index++;
    }
  }

  return results;
}


// <Graphviz dot={makeDot(messages)} options={{width: 1000, height: 1000, fit: true }} />


 // <Graphviz dot={dot} options={{width: 1000, height: 1000, fit: true }} />
 //      <input vale={index} onChange={e => setIndex(e.target.value)} type="range" min="0" max={dots.length - 1} step="1" list="steplist" />
 //      <datalist id="steplist">
 //        {range(hashes.length).map(x => <option key={x}>x</option>)}
 //      </datalist>
 //      <ul>
 //        {messages.map((x, i) => <li key={`${x.hash}-${i}`}>{x.hash}</li> )}
 //      </ul>

const blocksByEpochAccending = (array) => {
  const blocksByEpoch = sortBy(array, x => x.epoch);
  // console.log("by", blocksByEpoch)
  return partitionBy(blocksByEpoch, x => x.epoch);
}



const useAddListeners = (setSelected, deps) => {

  useEffect(() => {


    const titles = Array.from(document.getElementsByTagName("title"))
      .filter(x => x.innerHTML.includes("-"))
      .map(x => x.parentElement);

    let listeners = [];


    for (let node of titles) {
      let fn = (e) => {
          let id = `block-${node.children[0].innerHTML}`;
          let codeList = document.getElementById("codeList").parentElement;
          let blockOffset = document.getElementById(id)?.offsetTop;
          console.log(blockOffset)
          if (blockOffset) {
            document.getElementById(id).scrollIntoView({behavior: "smooth"})
            // codeList.scrollTo({top: blockOffset - 140, behavior: 'smooth'})
            setSelected(id)
          }
      }
      node.addEventListener("mouseover", fn);
      listeners.push({node, fn});
    }

    return () => {
      for (let {node, fn} of listeners) {
        node.removeEventListener("mouseover", fn)
      }
    }

  }, deps)
}

const reverse = (arr) => {
  const temp = [...arr];
  arr.reverse();
  return arr;
}



const DetailView = ({ blocks, name, setMethod }) => {

  const [rendered, setRendered] = useState(0);
  const [selected, setSelected] = useState(null);
  const [index, setIndex] = useState(0);
  // Epochs are backwards
  const [epoch, setEpoch] = useState(0)
  useAddListeners(setSelected, [rendered]);


  const blocksByIseq = groupBy(blocks, x => x.iseq_hash);
  const sortByCount = reverse(sortBy(Object.entries(blocksByIseq), x => x[1].length));
  const largestIseq = sortByCount[index][1];
  // console.log(messagesByIseq, sortByCount, largestIseq)
  const epochs = blocksByEpochAccending(largestIseq);




  const lastGroup = epochs[epochs.length - 1 - epoch] || [];


  const dot = useMemo(() => makeDot(lastGroup), [lastGroup]);


  return (

    <div>
      <div style={{backgroundColor: "#e8e9eb", height: 60, padding: 10}}>
        <div style={{backgroundColor: "white", display: "inline-block", marginRight: 10 }}>
          <h4 onClick={e => setMethod(null)} style={{cursor: "pointer",  padding: 10, margin: 0}}>❮</h4>
        </div>
        <div style={{backgroundColor: "white", display: "inline-block", }}>
          <h4 style={{padding: 10, margin: 0}}>{name}</h4>
        </div>
      </div>
      <div style={{display: "grid", gridTemplateColumns: "0.7fr 2.7fr 1fr", marginTop: 30, gap: 10}}>
        <div style={{ display: "flex", flexDirection: "column", height: "81vh" , overflow: "scroll", padding: 10}}>
          {sortByCount.map(([k, vals], i) => (
            <div onClick={e => {setIndex(i); setEpoch(0)}} style={{ backgroundColor: "white", height: 50, textAlign: "center", margin: 10, cursor: "pointer", boxShadow: index === i ? "4px 4px 4px 4px #e8e9eb" : "" }} key={k}>
              <p>Instruction Sequence {i + 1}</p>
            </div>
          ))}
        </div>
        <div style={{backgroundColor: "white", padding: 20 }}>
          <Graphviz onRender={() => setRendered(render => render + 1)} id={`graph-detail-${name}`} dot={dot} options={{width: 1000, height: 750, fit: true }} />

           <input value={epochs.length - 1 - epoch} onChange={e => setEpoch(epochs.length - 1 - e.target.value)} type="range" min="0" max={epochs.length -1} step="1" list="steplist" />
           <datalist id="steplist">
             {range(epochs.length - 1).map(x => <option key={x}>x</option>)}
           </datalist>
        </div>
        <div key={index} style={{height: "81vh", overflow: "scroll", marginRight: 20}}>
          <div  id="codeList">
            {lastGroup.filter(block => block.instructions.length !== 0).map((block, i) => {
              const id = `block-${block.hash}-${block.epoch}`;
              return (
                <div id={id} key={id} style={{backgroundColor: id === selected ? "#eee" : "white" , padding: "10px 20px 10px 20px", marginBottom: 10}}>
                  <pre>
                    {block.instructions.map(x => `${x.type === "comment" ? "#" : ""} ${x.value}`).join("\n")}
                  </pre>
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )

}


const Pill = ({ label, backgroundColor }) => {
  const pillColor = backgroundColor || "rgb(201 202 204)"
  return (
    <p style={{
        color:"white",
        marginLeft: 10, 
        border: `3px solid ${pillColor}`, 
        backgroundColor: pillColor, 
        borderRadius: 100, 
        width:50, 
        height:25, 
        textAlign: "center"
      }} >
        {label}
    </p>
  )
}




const CountBadges = ({ iseqs }) => {
  const toTake = iseqs.length > 3 ? 2 : 3;
  return (
    <div style={{display:"flex", marginLeft: "auto", marginRight: 20}}>
      {take(toTake, iseqs).map(([k, values]) => <Pill key={k} label={values.length} />)}
      {iseqs.length > 3 ? <Pill backgroundColor="rgb(142, 169, 223)" label={`+${iseqs.length - toTake}`} /> : null }
    </div>
  )
}



const ListView = ({ names, messagesByName, setName, cache, setCache }) => {


  return (
    <div style={{display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 1fr))", padding: 30,}}>
      {names.map((name, i) => {
        // TODO: I should just store state this way.
        const messages = messagesByName[name];
        const messagesByIseq = groupBy(messages, x => x.iseq_hash);
        const sortByCount = sortBy(Object.entries(messagesByIseq), x => x[1].length);
        const largestIseq = sortByCount[sortByCount.length - 1][1];
        // console.log(messagesByIseq, sortByCount, largestIseq)
        const epochs = blocksByEpochAccending(largestIseq);
        return (
          <div style={{margin: 20, boxShadow: "4px 4px 4px 4px #e8e9eb", backgroundColor: "#fff"}} onClick={(e) => setName(name)} key={name}>
            <div style={{borderBottom: "2px solid rgb(232, 233, 235)", display: "flex"}}>
              <h3 style={{padding: "0px 20px 10px 20px", maxWidth: 200}}>{messages[0].name}</h3>
              <CountBadges iseqs={sortByCount} />
            </div>
            <div style={{padding: 10}}>
                <Graphviz cache={cache} setCache={setCache} id={`graph-${name}-${i}`} dot={makeSimplifiedGraph(epochs[epochs.length - 1])} options={{width: 300, height: 300, fit: true }} />
            </div>
          </div>  
        )
      })}
    </div>
  )
}



const GraphvizPage = () => {


  const [cache, setCache] = useState({});
  const [name, setName] = useState(null);
  const webSocketMessages = useWebSocket();
  const [savedMessages, setSavedMessages] = useLocalForage("messages", []);
  const messages = savedMessages.concat(webSocketMessages);


  const messagesByName = useMemo(() => groupBy(messages, x => x.name), [messages])
  const names = Object.keys(messagesByName).filter(x => x !== "");
  names.sort();

  if (name !== null) {
    const blocks = messagesByName[name];
    return <DetailView name={name} blocks={blocks} setMethod={setName} />
  }

  return <ListView cache={cache} setCache={setCache} names={names} messagesByName={messagesByName} setName={setName} />
  
}

export default function Home() {

  return (
    <NoSSR>
      {process.browser && 
      <div>
        <div style={{padding: 10, boxShadow: "0 4px 4px 0 #e8e9eb", position: "fixed", width: "100%", backgroundColor: "#f9f9fb"}}>
          <div style={{margin: 20}}>
            <img src="yjit.png" style={{height: 35}} />
          </div>
        </div>
        <div style={{paddingTop: 100}}>
          <GraphvizPage />
        </div>
      </div>
      }
    </NoSSR>
  )
}
