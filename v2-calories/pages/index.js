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
      padding: 15,
      textAlign: "right"
    }}
  >
    {name}
    <div style={{ fontSize: 13, marginTop: -10 }}>{calories} cal</div>

    <style jsx>{`
      .entry:hover, .entry:focus {
        filter: brightness(120%);
        background-color: #272727;
      }


    `}
    </style>
  </div>
);

const Home = () => (
  <div>
    <Head>
      <link href="https://unpkg.com/superstylin@2.0.2/src/index.css" rel="stylesheet" />
      <title>Home</title>
    </Head>

    <h1>1200 Calories</h1>

    <Entry name="Burrito" calories="1150" first />
    <Entry name="Bowl" calories="850" />
    <Entry name="Cortado" calories="80" />
    <Entry name="Biscuit" calories="180" />
    <Entry name="Burrito" calories="1150" />
    <Entry name="Bowl" calories="850" />
    <Entry name="Cortado" calories="80" />
    <Entry name="Biscuit" calories="180" />
    <Entry name="Burrito" calories="1150" />
    <Entry name="Bowl" calories="850" />
    <Entry name="Cortado" calories="80" />
    <Entry name="Biscuit" calories="180" />
  </div>
)

export default Home
