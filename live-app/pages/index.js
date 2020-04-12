import useSWR, { SWRConfig, trigger, mutate } from "swr";
import { Suspense, useState, useEffect, useRef } from "react";
import { Editor, renderElementAsync } from "@jimmyhmiller/react-live";
import { useDebounce } from 'use-debounce';
import prettier from "prettier/standalone";
import parserBabel from "prettier/parser-babylon";
import CommandPalette from 'react-command-palette';

const formatCode = (code) => {
  return prettier.format(code, {
    parser: "babel",
    plugins: [parserBabel],
    printWidth: 66
  })
}

const formatCodeHandler = (code, setCode) => e => {
  // 70 is f for format. Not great for windows
  if (e.ctrlKey && e.keyCode === 70) {
    setCode(formatCode(code));
  }
};


const httpRequest = ({ method, body, url }) => {
  return fetch(url, {
    method: method,
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body)
  });
}

const api = {
  get: (url) => {
    return httpRequest({ method: "GET", url })
  },
  post: (url, { body }) => {
    return httpRequest({ method: "POST", body, url })
  },
  delete: (url) => {
    return httpRequest({ method: "DELETE", url })
  },
}


// https://github.com/gregberge/loadable-components/issues/322#issuecomment-553370417
const LoadingIndicatorWithDelay = ({ delay=300 }) => {
  const [showLoadingIndicator, setLoadingIndicatorVisibility] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setLoadingIndicatorVisibility(true), delay);
    return () => {
      window.clearTimeout(timer);
    };
  });

  return showLoadingIndicator ? "loading..." : null;
};

const editorColor = "#2a2f38"

const ViewingArea = ({ code }) => {
  const [Element, setElement] = useState(() => () => null);

  useEffect(() => {
    try {
      renderElementAsync(
        {
          code: `${code};\n\n render(Main);`,
          scope: { React, useState, useSWR, trigger, mutate, SWRConfig, api }
        },
        elem => setElement(_ => elem),
        e => console.error(e)
      );
    } catch (e) {
      console.error(e);
    }
  }, [code]);

  return <Element />;
};

const CreateRoute = () => {
  const [route, setRoute] = useState("");

  return (
    <div>
      Route:
      <input value={route} onChange={e => setRoute(e.target.value)} />
      <div>
        <button
          onClick={async () => {

            await httpRequest({ 
              url: "/admin/entity",
              method: "POST",
              body: {
                route,
                type: "endpoint",
                code: `(req, res) => res.send("Hello World")`
              }
            })
            trigger("/admin/entity?type=endpoint");
          }}>
          Create
      </button>
    </div>
  </div>
  )
}

const noop = () => {}

const useUpdateEntity = ({ endpoint, entity, code, beforeUpdate, afterUpdate }) => {
  const isFirstRun = useRef(true);
  useEffect(() => {

    // We don't need to update on the first render.
    if (isFirstRun.current) {
      isFirstRun.current = false;
      return;
    }

    const updateFunction = async () => {
      beforeUpdate({...entity, code})

      await httpRequest({
        url: endpoint,
        method: "POST",
        body: {
          ...entity,
          code,
        }
      })

      afterUpdate({...entity, code})
    }

    updateFunction();

  }, [code])
}

const deleteEntityHandler = ({ identifier, type }) => async e => {
  await httpRequest({
    url: `/admin/entity?type=${type}&identifier=${identifier}`,
    method: "DELETE",
  })

  // Probably want to be able to not trigger list?
  // Or maybe I want to modify the list locally?
  // Not sure.
  trigger(`/admin/entity?type=${type}`)
}

const EditorCard = ({ code, setCode, title, identifier, type, }) => {
  const cardBodyStyle = {
    backgroundColor: editorColor,
    width: 550,
    caretColor: "white",
    margin: 20,
    borderRadius: 5
  };

  const cardHeaderStyle = {
    color: "white",
    padding: 10,
    display: "flex",
    borderBottom: `2px solid ${editorColor}`,
    filter: "brightness(80%)",
    minHeight: 15
  };

  return (
    <div style={cardBodyStyle} onKeyUp={formatCodeHandler(code, setCode)}>
      <div style={cardHeaderStyle}>
        {title}
        <div onClick={deleteEntityHandler({identifier, type})} style={{marginLeft:"auto", cursor: "pointer"}}>X</div></div>
      <Editor
        padding={20}
        language="jsx"
        code={code}
        onValueChange={code => setCode(code)}
      />
    </div>
  );
};

const EntityEditor = ({ endpoint, type, identifier, beforeUpdate=noop, afterUpdate=noop, entity, titleFn }) => {
  const [code, setCode] = useState(entity.code);
  const [debouncedCode] = useDebounce(code, 300);
  useUpdateEntity({ endpoint, entity, code: debouncedCode, beforeUpdate, afterUpdate });

  return (
    <EditorCard type={type} identifier={identifier} code={code} setCode={setCode} title={titleFn(entity)} />
  )
}

const mutateComponentCode = ({ name, code }) => {
  mutate(`/admin/entity?identifier=${name}&type=component`, {name, code}, false)  
}

const ComponentEditor = ({ entity }) => {
  return (
    <EntityEditor
      endpoint="/admin/entity"
      type="component"
      identifier={entity.name}
      beforeUpdate={mutateComponentCode}
      entity={{...entity, type:"component"}}
      titleFn={entity => entity.name}
    />
  );
};

const RouteEditor = ({ entity }) => {
  return (
    <EntityEditor
      endpoint="/admin/entity"
      type="endpoint"
      identifier={entity.route}
      afterUpdate={() => trigger(`/api/${entity.route}`)}
      entity={{...entity, type:"endpoint"}}
      titleFn={entity => `/api/${entity.route}`}
    />
  )
}


const Commands = () => {
  const [commands, setCommands] = useState([{name: "Create Component", command: (...args) => console.log(args)}]);
  return (
    <CommandPalette
      showSpinnerOnSelect={false}
      onSelect={() => setCommands([{name: "Create Component Named _", command: () => {}}])}
      commands={commands} />
  );
}

const ListEntities = ({ keyFn, endpoint, Editor }) => {
  const { data } = useSWR(endpoint, {dedupingInterval: 200});
  return (
    <div>
      {data.map(entity => <Editor key={keyFn(entity)} entity={entity} />)}
    </div>
  );
}

const ViewMainComponent = () => {
  const { data: {name, code} } = useSWR("/admin/entity?identifier=Main&type=component");
  return <ViewingArea code={code} />
}

const ListCollections = () => {
  const { data: { collections } } = useSWR("/admin/collection");
  const [name, setName] = useState("");

  return (
    <>
      <ul>
        {collections.map(x => <li key={x.name}>{x.name}</li>)}
      </ul>
      <input 
        placeholder="collection" 
        value={name} 
        onChange={(e) => setName(e.target.value)} />
      <button
        onClick={async () => {
          await httpRequest({
            url: "/admin/collection",
            method: "POST",
            body: {
              name
            }
          })

          mutate("/admin/collection", {collections: collections.concat([{name}])});
          setName("");
        }}
      >Add Collection</button>
    </>
  )
}

const Index = () => {
  return (
    <SWRConfig
      value={{
        suspense: process.browser,
        refreshInterval: 0,
        dedupingInterval: 0,
        fetcher: (...args) => fetch(...args).then(res => res.json())
      }}
    >
    <style jsx global> {`
       body {
         font-family: helvetica;
       }
    `}
    </style>
      {/*ugliness because suspense doesn't support server side rendering*/}
      {process.browser ? (
        <Suspense fallback={<LoadingIndicatorWithDelay />}>
          <div style={{display: "flex", flexDirection: "row"}}>
            <div style={{width: "45vw", height: "95vh", overflow: "scroll"}}>
               <ListCollections />
               <ListEntities 
                 keyFn={e => e.route} 
                 endpoint={"/admin/entity?type=endpoint"} 
                 Editor={RouteEditor} />
               <ListEntities 
                 keyFn={e => e.name} 
                 endpoint={"/admin/entity?type=component"} 
                 Editor={ComponentEditor} />
              <CreateRoute />
             </div>
            <div style={{width: "45vw", height: "95vh", padding: 20}}>
              <ViewMainComponent />
            </div>
          </div>
        </Suspense>
      ) : (
        null
      )}
    </SWRConfig>
  );
};

export default Index;