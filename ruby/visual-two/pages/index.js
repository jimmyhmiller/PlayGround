import { useState, useEffect, useRef, useMemo } from 'react';
import dynamic from 'next/dynamic';
import NoSSR from '@mpth/react-no-ssr';
import localforage from 'localforage';



// TODO: Make svg cache

// console.log(d3)

// const Graphviz = dynamic(() => import('graphviz-react'), { ssr: false });

const Graphviz = ({id, dot, options: {width, height}, onRender}) => {
  let selector = id.replace(/[^A-Za-z\-0-9]/g, "")
  useEffect(() => {


    window["@hpcc-js/wasm"].graphviz.layout(dot, "svg", "dot").then(svg => {
      const div = document.getElementById(selector);
      if (!div) {
        console.error(`no selector ${selector}`)
        return;
      }
      div.innerHTML = svg;
      div.children[0].setAttribute("width", `${width}px`)
      div.children[0].setAttribute("height", `${height}px`)
      onRender && onRender()
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

  useEffect(() => {
    (async () => {
      const item = await localforage.getItem(key);
      if (item) {
        setStoredValue(item);
      }
    })()
  }, [])


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
  let results = [[array[0]]];
  let lastValue = f(array[0]);
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




const DetailView = ({ blocks, name, setIndex }) => {

  const [rendered, setRendered] = useState(false);
  const [selected, setSelected] = useState(null);

  useEffect(() => {


    console.log("adding handlers")
    const titles = Array.from(document.getElementsByTagName("title"))
      .filter(x => x.innerHTML.includes("-"))
      .map(x => x.parentElement);

    let listeners = [];


    for (let node of titles) {
      let fn = (e) => {
          let id = `block-${node.children[0].innerHTML}`;
          let codeList = document.getElementById("codeList").parentElement;
          let blockOffset = document.getElementById(id)?.offsetTop;
          if (blockOffset) {
            codeList.scrollTo({top: blockOffset - 70, behavior: 'smooth'})
            setSelected(id)
          }
      }
      node.addEventListener("mouseover", fn);
      listeners.push({node, fn});
    }

    return () => {
      console.log("removing handlers")
      for (let {node, fn} of listeners) {
        node.removeEventListener("mouseover", fn)
      }
    }

  }, [rendered])

  const blockGroups = blocksByEpochAccending(blocks);
  const lastGroup = blockGroups[blockGroups.length - 1];


  const dot = useMemo(() => makeDot(lastGroup), [lastGroup]);


  return (
    <div style={{display: "grid", gridTemplateColumns: "3fr 1fr"}}>
      <div>
        <h5 onClick={e => setIndex(null)}>{name}</h5>
        <Graphviz onRender={() => setRendered(true)} id={`graph-detail-${name}`} dot={dot} options={{width: 1000, height: 1000, fit: true }} />
      </div>
      <div style={{height: "99vh", overflow: "scroll"}}>
        <div id="codeList">
          {lastGroup.map((block, i) => {
            const id = `block-${block.hash}-${block.epoch}`;
            return (
              <div id={id} key={id} style={{backgroundColor: id === selected ? "#eee" : "white" }}>
                <pre>
                  {block.instructions.map(x => `${x.type === "comment" ? "#" : ""} ${x.value}`).join("\n")}
                </pre>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )

}


const ListView = ({ names, messagesByName, setIndex, setCache }) => {


  return (
    <div style={{display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 1fr))", padding: 30,}}>
      {names.map((name, i) => {
        const messages = messagesByName[name];
        const epochs = blocksByEpochAccending(messages);
        return (
          <div style={{margin: 20, boxShadow: "4px 4px 4px 4px #e8e9eb", backgroundColor: "#fff"}} onClick={(e) => setIndex(i)} key={name}>
            <div style={{borderBottom: "2px solid rgb(232, 233, 235)"}}>
              <h3 style={{padding: "0px 20px 10px 20px", }}>{messages[0].name}</h3>
            </div>
            <div style={{padding: 10}}>
              <Graphviz id={`graph-${name}-${i}`} dot={makeSimplifiedGraph(epochs[epochs.length - 1])} options={{width: 300, height: 300, fit: true }} />
            </div>
          </div>  
        )
      })}
    </div>
  )
}



const GraphvizPage = () => {


  const [index, setIndex] = useState(null);
  const webSocketMessages = useWebSocket();
  const [savedMessages, setSavedMessages] = useLocalForage("messages", []);
  const messages = savedMessages.concat(webSocketMessages);


  const messagesByName = useMemo(() => groupBy(messages, x => x.name), [messages])
  const names = Object.keys(messagesByName).filter(x => x !== "");
  names.sort();

  if (index !== null) {
    const name = names[index];
    const blocks = messagesByName[names[index]];
    return <DetailView name={name} blocks={blocks} setIndex={setIndex} />
  }

  return <ListView names={names} messagesByName={messagesByName} setIndex={setIndex} />
  
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
