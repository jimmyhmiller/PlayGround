import React, { Suspense, useState } from 'react'
import Link from 'next/link'
import Head from 'next/head'
import useSWR, { SWRConfig, trigger, mutate } from "swr";
// in browser we don't need time zone stuff
import { startOfToday, format as formatDate } from "date-fns";



const today = () => formatDate(startOfToday(), "yyyy-MM-dd");
console.log(today())

const httpRequest = ({ method, body, url }) => {
  return fetch(url, {
    method: method,
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body)
  });
}


const Entry = ({ name, calories, first }) => (
  <div>
    <button
      onClick={async () => {

        await httpRequest({ 
          url: "/api/entry",
          method: "POST",
          body: {
            calories,
            name,
            date: today()
          }
        })
        trigger("/api/entry?summary=true");
      }}
      className="entry"
      style={{
        fontSize: 26,
        borderTop: first ? "1px solid #343334" : "none",
        borderBottom: "1px solid #343334",
        padding: "10px 15px 10px 15px",
        textAlign: "right"
      }}
    >
      {name}
      <div style={{ fontSize: 13, marginTop: -10 }}>{calories} cal</div>

    {/*Think about light and dark theme, or just make it all dark*/}
      <style jsx>{`

        .entry {
          box-shadow: 5px 5px 10px #1d1d1d, 
              -5px -5px 10px #313131;
          margin-top:10px;
          width: 100%;
          background-color: #272727;
          border: none;
          color: #c1c1c1;
          line-height: 30px;
        }

        .entry:active {
          box-shadow: 5px 5px 10px #313131, 
              -5px -5px 10px #1d1d1d;
        }


      `}
      </style>
    </button>
  </div>
);

const subtractRemaining = ({ summary, calories }) => {
  return {
    summary: {
      ...summary,
      remaining: summary.remaining - calories
    }
  }
}

const AddItem = ({ calories, summary, setCalories }) => {
  return (
    <div
      onClick={async () => {
        if (!calories) {
          return;
        }

        await httpRequest({ 
          url: "/api/entry",
          method: "POST",
          body: {
            calories: parseInt(calories, 10),
            name: "food",
            date: today(),
          }
        })

        mutate("/api/entry?summary=true", subtractRemaining({ summary, calories}));
        setCalories("");
      }}
      className="add-item"
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        position:"fixed",
        bottom: 20,
        right: 20,
        backgroundColor: "#272727",
        width: 70,
        height: 70,
        zIndex:99999,
        borderRadius: 19,
        textShadow: "0px 1px 0px #2e2e2e", 
        // background: "linear-gradient(145deg, #f8fef7, #d1d5d0)",
        
        // color: "black",
        fontSize: 90,
        fontWeight: 400,
      }}
    ><span>+</span>

    <style jsx>{`

      .add-item {
        box-shadow: 5px 5px 6px #202020, -5px -5px 6px #2e2e2e;
        -webkit-text-stroke: 2px #c1c1c1;
        -webkit-text-fill-color: #272727;
      }

      .add-item:hover {
        box-shadow: 5px 5px 10px #151515, -5px -5px 10px #393939;
      }

      .add-item:active {
        box-shadow: -5px -5px 6px #202020, 5px 5px 6px #2e2e2e;
        -webkit-text-stroke: none;
        -webkit-text-fill-color: #c1c1c1;
      }


    `}
    </style>

    </div>
  )
}

const AddEntry = ({ summary }) => {
  const [calories, setCalories] = useState("");
  return (
    <>
      <Entry name="Bowl" calories={850} first />
      <Entry name="Cortado" calories={100} />
      <Entry name="Biscuit" calories={185} />
      <Entry name="1 Mile Walk" calories={-100} />
      {/*Ugly*/}
      <input
        value={calories}
        onChange={e => setCalories(e.target.value)} 
        style={{marginTop: 30, width: "100%", borderRadius: 0}} 
        type="number" 
        placeholder="Calories" />
      <AddItem calories={calories} setCalories={setCalories} summary={summary} />
    </>
  )
}

// I stopped using this for a while the 47000 is a little hack to reset things.
const Summary = ({ summary }) => {
  return (
    <ul>
      <li>Remaining: {summary["remaining"]}</li>
      <li>Extra One Pound: {summary["extraOnePound"] - 47000}</li>
      <li>Extra Two Pounds: {summary["extraTwoPounds"]}</li>
      <li>Pounds: {summary["pounds"]}</li>
      <li>Pounds By Weeks: {summary["projectedLoss"]}</li>
      <li>Days: {summary["days"]}</li>
      <li>Weeks: {summary["weeks"]}</li>
      <li>Total: {summary["total"]}</li>
      <li>Daily: {summary["daily"]}</li>
    </ul>
  )
}


const Main = () => {
  const {data : { summary }} = useSWR("/api/entry?summary=true");
  const [showSummary, setShowSummary] = useState(false);
  return (
    <div>
      <h1 onClick={() => setShowSummary(!showSummary)}>{summary.remaining} Calories</h1>
      {showSummary ? <Summary summary={summary} /> : <AddEntry summary={summary} />}
    </div>
  )
}


const App = () => {
  return (
    <SWRConfig
      value={{
        suspense: process.browser,
        refreshInterval: 0,
        fetcher: (...args) => fetch(...args).then(res => res.json())
      }}
    >
      <Head>
        <link href="https://unpkg.com/superstylin@2.0.2/src/index.css" rel="stylesheet" />
        <title>Calories</title>

        <link rel="apple-touch-icon" sizes="180x180" href="/static/apple-touch-icon.png" />
        <link rel="icon" type="image/png" sizes="32x32" href="/static/favicon-32x32.png" />
        <link rel="icon" type="image/png" sizes="16x16" href="/static/favicon-16x16.png" />
        <link rel="manifest" href="/static/site.webmanifest" />
      </Head>

      <style jsx global>{`
        body {
          color: #c1c1c1;
          font-family: 'helvetica', arial;
          margin: 20px;
        }

        h1 {
          margin-top: 0;
          text-shadow: 
                5px 5px 10px #101010, 
              -5px -5px 10px #3e3e3e;
        }

      `}</style>
      {process.browser ? 
        <Suspense fallback={<p></p>}>
          <Main /> 
        </Suspense>
        : null
      }
    </SWRConfig>
  )
}



export default App
