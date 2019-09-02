import React, { useState } from 'react'
import Link from 'next/link'
import Head from 'next/head'



const distinct = (coll) => [...new Set(coll)];

const extractSingleLetterWords = (text) => {
  return distinct(
    text
      .split(/ |\.|\n/)
      .filter(x => x.match(/^([b-z]|[a-z]'+|[a-z][0-9]|[A-Z]+)$/))
  )
}

const extractStartWhitespace = (text) => console.log(text.match(/(\W*).*/)) ||  text.match(/(\W*).*/)[1]

const fillSingleLetterWords = (text, variables) => {
  return text
      .split(/(?= |\. |\n)/)
      .map(x => {
        if (x.trim().match(/^([b-z]|[a-z]'+|[a-z][0-9]|[A-Z]+)$/)) {
          return (variables[x.trim()] && extractStartWhitespace(x) + variables[x.trim()] ) || x;
        }
        return x;
      }).join("")
}


const useInput = (initialState) => {
  const [value, setValue] = useState(initialState);
  return {
    value, 
    onChange: (e) => setValue(e.target.value)
  }
}

const Variable = ({ name, value, onChange }) => {

  return (
    <p>
      <span style={{paddingRight: 20}}>{name}:</span>
      <input
        value={value}
        onChange={onChange}
        type="text" style={{fontSize: 16}} />
    </p>
  )
}

const useVariables = () => {
  const [variables, setVariables] = useState({});

  const changeSelf = (name) => (e) => {
    setVariables({
      ...variables,
      [name]: e.target.value
    })
  }

  return [variables, changeSelf]
}


const Index = () => {

  const proposition = useInput("");
  const [variables, setVariables] = useVariables();

  return (
    <div>

      <style jsx>{`
        .highlight {
          background-color: #f6f6f6;
        }

        @media (prefers-color-scheme: dark) {
          .highlight {
            background-color: #595859;
          }
        }
      `}
      </style>
      <Head>
        <title>Proposition Filler</title>
        <link href="https://unpkg.com/superstylin@2.0.2/src/index.css" rel="stylesheet" />
      </Head>

      <h1>Proposition Filler</h1>

      <hr style={{margin:10}} />

      <p>Proposition:</p>
      <textarea 
        {...proposition}
        style={{fontSize: 16, maxWidth: "85vw", width: "500px", height: 100}}  />

      <p>Variables:</p>
      {extractSingleLetterWords(proposition.value)
        .map(name => <Variable key={name} name={name} value={variables[name] || ""} onChange={setVariables(name)} />)
      }
      <div
        className="highlight"
        style={{
          borderRadius: 10,
          padding: "0.5rem",
          maxWidth: "85vw",
          width: "500px",
          minHeight: 100,
          fontSize: 16,
          fontFamily: "-apple-system",
        }}
      >
        <pre style={{whiteSpace: "pre-wrap"}}>
          {fillSingleLetterWords(proposition.value, variables)}
        </pre>
      </div>


    </div>
  )
}

export default Index
