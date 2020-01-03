import useSWR, { SWRConfig, trigger, mutate } from "swr";
import { Suspense, useState, useEffect } from "react";
import { Editor, renderElementAsync } from "react-live";
import { useDebounce } from 'use-debounce';
import prettier from "prettier/standalone";
import parserBabel from "prettier/parser-babylon";

const formatCode = (code) => {
  return prettier.format(code, {
    parser: "babel",
    plugins: [parserBabel],
    printWidth: 58
  })
}


// https://github.com/gregberge/loadable-components/issues/322#issuecomment-553370417
const LoadingIndicatorWithDelay = ({ delay=200 }) => {
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
          scope: { React, useState, useSWR }
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


const Route = ({ code: initialCode, route }) => {
  const [code, setCode] = useState(initialCode);
  const [debouncedCode] = useDebounce(code, 300);
  useEffect(() => {
    const updateFunction = async () => {
      await fetch("/admin/update-route", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          route,
          code: debouncedCode,
        })
      });
      console.log("Triggering")
      trigger(`/api/${route}`);
    }

    updateFunction();

  }, [route, debouncedCode])

  return (
      <div style={{backgroundColor: editorColor, width: 500, caretColor: "white", margin:20, borderRadius: 5}}
           onKeyUp={e => { 
             if (e.ctrlKey && e.keyCode === 70) {
              setCode(formatCode(code))
            }
           }}>
        <div style={{color: "white", padding:10, borderBottom: `2px solid ${editorColor}`, filter: "brightness(80%)", minHeight: 15}}>
          /api/{route}
        </div>
        <Editor
          padding={20}
          language="jsx"
          code={code}
          // onBlur={() => prettifyCode({ name })}
          onValueChange={(code) => setCode(code)}
        />
    </div>
  );
};

const Routes = () => {
  const { data: routes } = useSWR("/admin/routes");
  return (
    <div>
      {routes.map(r => <Route key={r.route} {...r} />)}
    </div>
  );
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
            await fetch("/admin/update-route", {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({
                route,
                code: `(req, res) => res.send("Hello World")`
              })
            });
            trigger("/admin/routes");
          }}>
          Create
      </button>
    </div>
  </div>
  )
}

// Should probably get rid of duplicated code

const Component = ({ code: initialCode, name }) => {
  const [code, setCode] = useState(initialCode);
  const [debouncedCode] = useDebounce(code, 300);
  useEffect(() => {
    const updateFunction = async () => {
      mutate(`/admin/component?name=${name}`, {name, code: debouncedCode}, false)  
      await fetch("/admin/update-component", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          name,
          code: debouncedCode,
        })
      });
      
      // trigger(`/admin/component?name={name}`);
    }

    updateFunction();

  }, [name, debouncedCode])

  return (
      <div style={{backgroundColor: editorColor, width: 500, caretColor: "white", margin:20, borderRadius: 5}}
           onKeyUp={e => { 
             if (e.ctrlKey && e.keyCode === 70) {
              setCode(formatCode(code))
            }
           }}>


        <div style={{color: "white", padding:10, borderBottom: `2px solid ${editorColor}`, filter: "brightness(80%)", minHeight: 15}}>
          {name}
        </div>
        <Editor
          padding={20}
          language="jsx"
          code={code}
          // onBlur={() => prettifyCode({ name })}
          onValueChange={(code) => setCode(code)}
        />
    </div>
  );
};

const Components = () => {
  const { data: components } = useSWR("/admin/components");
  return (
    <div>
      {components.map(c => <Component key={c.name} {...c} />)}
    </div>
  );
};


const CreateComponent = () => {
  const [name, setName] = useState("");

  return (
    <div>
      Component Name:
      <input value={name} onChange={e => setName(e.target.value)} />
      <div>
        <button
          onClick={async () => {
            await fetch("/admin/update-component", {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({
                name,
                code: `(req, res) => res.send("Hello World")`
              })
            });
            trigger("/admin/components");
          }}>
          Create
      </button>
    </div>
  </div>
  )
}

const ViewMainComponent = () => {
  const { data: {name, code} } = useSWR("/admin/component?name=Main");
  return <ViewingArea code={code} />
}

const Index = () => {
  return (
    <SWRConfig
      value={{
        suspense: process.browser,
        refreshInterval: 0,
        dedupingInterval: 10,
        fetcher: (...args) => fetch(...args).then(res => res.json())
      }}
    >
      {/*ugliness because suspense doesn't support server side rendering*/}
      {process.browser ? (
        <Suspense fallback={<LoadingIndicatorWithDelay />}>
          <div style={{display: "flex", flexDirection: "row"}}>
            <div style={{width: "45vw", height: "95vh", overflow: "scroll"}}>
              <Routes />
              <Components />
              <CreateRoute />
              {/*<CreateComponent />*/}
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