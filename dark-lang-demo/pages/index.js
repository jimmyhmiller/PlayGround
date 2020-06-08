import Head from 'next/head'
import { useState } from 'react';

// Need to add a next button and a way to transition between steps

const Step = ({ title, text, code, onClick, onNext, hasNext, step, onPrevious }) => {
  const [result, setResult] = useState();
  const [success, setSuccess] = useState(false);
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
            display: flex;
            flex-direction: column;
          }
          button:disabled {
            cursor: default;
            opacity: 0.3;
          }

        `}</style>
        <h1>{title}</h1>
        <p>{text}</p>
        <code><pre>
          {code}
        </pre></code>
        <div>
          <button onClick={onClick({setResult, setSuccess})}>Make Request</button>
        </div>
        {result && 
          <blockquote>
           {result}
           </blockquote>
        }
     
      <div style={{display: "flex", marginTop:30, justifyContent: "flex-end"}}>
        <button disabled={step === 0} onClick={onPrevious}>Previous</button>
        <button disabled={!(success && hasNext)} onClick={onNext}>Next</button>
      </div>
      </article>
    </div>
  )
}

const HelloWorld = ({ onNext, hasNext, onPrevious, step }) => {
  const [result, setResult] = useState();

  return (
    <Step
      step={step}
      hasNext={hasNext}
      onNext={onNext}
      onPrevious={onPrevious}
      title="Dark Lang Demo"
      text="Welcome to a Dark Lang Demo. To begin we are going to make simple hello world dark route."
      code={`GET /hello_world => "hello world"`}
      onClick={({setResult, setSuccess}) => async () => {
          setResult("");
          const response = await fetch("/api/hello_world");
          const body = await response.text();
          if (body.toLowerCase() === "hello world") {
            setResult("Success!")
            setSuccess(true)
            return;
          }
          setResult(`Expected "hello world" got "${body}" with status ${response.status}`)
        }
      }
    />
  )
}

const Greet = ({ onNext, hasNext, onPrevious, step }) => {
  const [result, setResult] = useState();
  const [name, setName] = useState("");

  return (
    <Step
      step={step}
      hasNext={hasNext}
      onNext={onNext}
      onPrevious={onPrevious}
      title="Greeting People"
      text={
        <div>
          <p>We are now going to make an endpoint that will accept the name as a url parameter and greet that person</p>
          <input type="text" placeholder="name" value={name} onChange={e => setName(e.target.value)} />
        </div>
      }
      code={`GET /greet/${name === "" ? ":name" : name} => "hello ${name === "" ? ":name" : name}"`}
      onClick={({setResult, setSuccess}) => async () => {
          setResult("");
          const response = await fetch(`/api/greet/${name}`);
          const body = await response.text();
          if (body.toLowerCase() === `hello ${name}`) {
            setResult("Success!")
            setSuccess(true)
            return;
          }
          setResult(`Expected "hello world" got "${body}" with status ${response.status}`)
        }
      }
    />
  )
}


const steps = [HelloWorld, Greet]

const Demo = () => {
  const [step, setStep] = useState(0);
  const Component = steps[step];
  return (
    <div className="container">
      <Head>
        <title>Create Next App</title>
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <Component 
        step={step}
        hasNext={step < steps.length - 1}
        onNext={() => setStep(step => step + 1)}
        onPrevious={() => setStep(step => step - 1)} />
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

export default Demo
