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
            max-width:700px;
          }

          article {
            width: 90vw;
            max-height: 80vh;
            max-width: 90vw;
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
          setResult(`Expected "hello ${name}" got "${body}" with status ${response.status}`)
        }
      }
    />
  )
}

const Echo = ({ onNext, hasNext, onPrevious, step }) => {
  const [result, setResult] = useState();

  return (
    <Step
      step={step}
      hasNext={hasNext}
      onNext={onNext}
      onPrevious={onPrevious}
      title="Echoing a POST"
      text={
        <div>
          <p>We are going to now try out a POST method with some JSON</p>
        </div>
      }
      code={`POST /echo => Whatever was passed in the body as json`}
      onClick={({setResult, setSuccess}) => async () => {
          let message = {
            pleaseEchoThisBack: "Echo!"
          }
          setResult("");
          const response = await fetch(`/api/echo`, {
            method: "POST",
            body: JSON.stringify(message),
            headers: {
              "Content-Type": "application/json"
            }
          });
          if (!(response.headers.get('content-type') || "").includes("application/json")) {
            setResult(`Wrong Content-Type: Expected application/json got ${response.headers.get('content-type')}`)
            return;
          }
          const body = await response.json();
          if (JSON.stringify(body) === JSON.stringify(message)) {
            setResult("Success!")
            setSuccess(true)
            return;
          }
          setResult(`Expected ${JSON.stringify(message)} got ${JSON.stringify(body)} with status ${response.status}`)
        }
      }
    />
  )
}

const PostMessage = ({ onNext, hasNext, onPrevious, step }) => {
  const [result, setResult] = useState();
  const [key, setKey] = useState("");
  const [message, setMessage] = useState("");

  return (
    <Step
      step={step}
      hasNext={hasNext}
      onNext={onNext}
      onPrevious={onPrevious}
      title="Saving Some Data"
      text={
        <div>
          <p>Now we want to be able to send a message, with a key and get that message back.</p>
          <input type="text" placeholder="key" value={key} onChange={e => setKey(e.target.value)} />
          <input type="text" placeholder="message" value={message} onChange={e => setMessage(e.target.value)} />
        </div>
      }
      code={
`POST /messages {key, message} => {success: true}
GET /messages/:key => { message }`}
      onClick={({setResult, setSuccess}) => async () => {
          setResult("");
          const postResponse = await fetch(`/api/messages`, {
            method: "POST",
            body: JSON.stringify({key, message}),
            headers: {
              "Content-Type": "application/json"
            }
          });
          const postBody = await postResponse.json();

          if (postBody.success !== true) {
            setResult(`Expected {success: true} got ${postBody} with status ${response.status}`)
            return;
          }

          const response = await fetch(`/api/messages/${key}`);
          if (!(response.headers.get('content-type') || "").includes("application/json")) {
            setResult(`Wrong Content-Type: Expected application/json got ${response.headers.get('content-type')}`)
            return;
          }
          const body = await response.json();
          if (body.message === message) {
            setResult("Success!")
            setSuccess(true)
            return;
          }
          setResult(`Expected ${message} got ${JSON.stringify(body)} with status ${response.status}`)
        }
      }
    />
  )
}

const GetMessages = ({ onNext, hasNext, onPrevious, step }) => {
  const [result, setResult] = useState();
  const [messages, setMessages] = useState([]);
  return (
    <Step
      step={step}
      hasNext={hasNext}
      onNext={onNext}
      onPrevious={onPrevious}
      title="Get Messages"
      text={
        <div>
          <p>Now we want to get a full list of messages</p>
          <ul>
            {messages.map((message, i) => {
              return <li key={i}>{message}</li>
            })}
          </ul>
        </div>
      }
      code={`GET /messages => {messages: [...]}`}
      onClick={({setResult, setSuccess}) => async () => {
          setResult("");

          const response = await fetch(`/api/messages`);
          if (!(response.headers.get('content-type') || "").includes("application/json")) {
            setResult(`Wrong Content-Type: Expected application/json got ${response.headers.get('content-type')}`)
            return;
          }
          const body = await response.json();
          if (body.messages && body.messages.length) {
            setMessages(body.messages)
            setResult("Success!")
            setSuccess(true)
            return;
          }
          setResult(`Expected an array of messages got ${JSON.stringify(body)} with status ${response.status}`)
        }
      }
    />
  )
}

// Need to demo feature flags.
// Maybe I do that by changing this route return the key as well?
// I need to demo workers, pusher seems like a good option?
// Or I could just do something in the background with polling?
// Maybe I just have some computed property I am constantly polling for.
// That would definitely be easier.

const steps = [HelloWorld, Greet, Echo, PostMessage, GetMessages]

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
