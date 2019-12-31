import useSWR, { SWRConfig, trigger } from "swr";
import { Suspense, useState, useEffect } from "react";

// https://github.com/gregberge/loadable-components/issues/322#issuecomment-553370417
const LoadingIndicatorWithDelay = ({ delay=200 }) => {
  const [showLoadingIndicator, setLoadingIndicatorVisibility] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setLoadingIndicatorVisibility(true), delay);
    return () => {
      window.clearTimeout(timer);
    };
  });

  return showLoadingIndicator ? <LoadingIndicator /> : null;
};


const Route = ({ code: initialCode, route }) => {
  const [code, setCode] = useState(initialCode);

  return (
    <>
      <h2>Route: /api/{route}</h2>
      <textarea
        value={code}
        onChange={e => setCode(e.target.value)}
        rows={10}
        cols={80}
      />
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
                code
              })
            });
          }}
        >
          Update
        </button>
      </div>
    </>
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


const Index = () => {
  return (
    <SWRConfig
      value={{
        suspense: process.browser,
        refreshInterval: 0,
        fetcher: (...args) => fetch(...args).then(res => res.json())
      }}
    >
      {/*ugliness because suspense doesn't support server side rendering*/}
      {process.browser ? (
        <Suspense fallback={<LoadingIndicatorWithDelay />}>
          <Routes />
          <CreateRoute />
        </Suspense>
      ) : (
        null
      )}
    </SWRConfig>
  );
};

export default Index;