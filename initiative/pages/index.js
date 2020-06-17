import { useState } from 'react';
import Head from 'next/head'
import { setIn } from 'zaphod/compat'
import orderBy from 'lodash/orderBy';


const Tracker = () => {
  const [name, setName] = useState("")
  const [players, setPlayers] = useState([])
  return (
    <>
      <form onSubmit={e => e.preventDefault()}>
      <input
        value={name}
        onChange={e => setName(e.target.value)}
        placeholder="Name" />
      <button type="submit" onClick={e => {
        setPlayers(players => players.concat([{name: name, initiative: ""}]));
        setName("");
      }}>
        Add
      </button>
      </form>

      <table>
        <thead>
          <tr>
            <th>Name</th>
            <th>Initiative</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {players.map(({name, initative}, i) => 
            <tr key={name}>
              <td>{name}</td>
              <td>
                <input
                  value={initative}
                  type="number"
                  style={{width: 30, marginLeft: 10}}
                  onChange={e => {
                    e.persist();
                    return setPlayers(players => setIn(players, [i, "initiative"], e.target.value))
                  }
                }/>
              </td>
              <td
                onClick={_ => setPlayers(players => players.filter(p => p.name != name))}
                style={{color: "red", cursor: "pointer"}}>
                ✖
              </td>
            </tr>)}
          </tbody>
      </table>
      <button onClick={_ => {
        setPlayers(players => orderBy(players, "initiative", "desc"))
      }}>Sort</button>
    </>
  )
}


const App = () => {
  return (
    <div className="container">
      <Head>
        <title>Initiative Tracker</title>
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <Tracker />

    </div>
  )
}

export default App;
