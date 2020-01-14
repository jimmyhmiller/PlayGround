import React from 'react'
import Link from 'next/link'
import Head from 'next/head'



const Entry = ({ name, calories, first }) => (
  <div
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
        margin-top:30px;
      }

      .entry:active {
        box-shadow: 5px 5px 10px #313131, 
            -5px -5px 10px #1d1d1d;
      }


    `}
    </style>
  </div>
);

const AddItem = () => {
  return (
    <div
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

const Home = () => (
  <div>
    <Head>
      <link href="https://unpkg.com/superstylin@2.0.2/src/index.css" rel="stylesheet" />
      <title>Home</title>
    </Head>

    <style jsx global>{`
      body {
        color: #c1c1c1;
        font-family: 'helvetica', arial;
      }

      h1 {
        text-shadow: 
              5px 5px 10px #101010, 
            -5px -5px 10px #3e3e3e;
      }

    `}</style>

    <h1>1200 Calories</h1>

    <Entry name="Bowl" calories="700" first />
    <Entry name="Cortado" calories="80" />
    <Entry name="Biscuit" calories="250" />
    <AddItem />
  </div>
)

export default Home
