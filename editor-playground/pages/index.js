import React from "react";
import Head from "next/head";

import { LiveProvider, LiveEditor, LiveError, LivePreview, Editor } from "react-live";

const Home = () => (
  <div>

    <style jsx global>{`
      body {
        background-color: #42374a
      }
    `}
    </style>
    <Head>
      <title>Home</title>
    </Head>


  <Editor
    language="jsx"
    style={{background: "rgb(50, 42, 56) none repeat scroll 0% 0%"}} 
    code="<strong>Hello World!</strong>" />

  <p />

  <Editor
    language="jsx"
    style={{background: "rgb(42, 47, 56) none repeat scroll 0% 0%"}} 
    code="<strong>Hello World!</strong>" />


  <p />

  <Editor
    language="jsx"
    style={{background: "rgb(56, 42, 42) none repeat scroll 0% 0%"}} 
    code="<strong>Hello World!</strong>" />






  </div>
);






export default Home;