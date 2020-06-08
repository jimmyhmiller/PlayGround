import Head from 'next/head'
import { useState } from 'react';

// Need to add a next button and a way to transition between steps

const Step = ({ title, text, code, onClick }) => {
  const [result, setResult] = useState();
  return (
    <div>
      <article>
        <style jsx>{`
          blockquote {
            max-height: 300px;
            overflow: scroll;
          }

          article {
            min-height: 400px;
          }

        `}</style>
        <h1>{title}</h1>
        <p>{text}</p>
        <code><pre>
          {code}
        </pre></code>
        <button onClick={onClick(setResult)}>Make Request</button>
        {result && 
          <blockquote>
           {result}
           </blockquote>
        }
      </article>
    </div>
  )
}

const HelloWorld = () => {
  const [result, setResult] = useState();

  return (
    <Step
      title="Dark Lang Demo"
      text="Welcome to a Dark Lang Demo. To begin we are going to make simple hello world dark route."
      code={`GET /hello_world => "hello world"`}
      onClick={(setResult) => async () => {
          setResult("");
          const response = await fetch("/api/hello_world");
          const body = await response.text();
          if (body.toLowerCase() === "hello world") {
            setResult("Success!")
            return;
          }
          setResult(`Expected "hello world" got "${body}" with status ${response.status}`)
        }
      }
    />
  )
}



export default function Home() {
  return (
    <div className="container">
      <Head>
        <title>Create Next App</title>
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <HelloWorld />
      <style jsx>{`
        .container {
          display: flex;
          align-items: center;
          justify-content: center;
          height: 100vh;
          width: 100vw;
        }
  
      `}</style>
    </div>
  )
}
