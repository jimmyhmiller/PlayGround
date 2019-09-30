import React, { useState, useEffect } from 'react';
import { Editor } from "react-live";
import prettier from "prettier/standalone";
import parserBabel from "prettier/parser-babylon";
import './App.css';
import { channels } from '../shared/constants';
const { ipcRenderer } = window; 


// Considered using monaco editor. But only one theme limitation is not great

// Need different theme
// Also not sure about these colors
// "#914b1f", "#a88a3f", "#47634d"

const colors = ["#2a2f38", "#322a38", "#022b3a", "#2a382d", "#382a2a" ]


// Next steps:
// ===============
// Set up reducers
// Set up export
// Set up filter
// Set up meta blocks

// Ideas
// ===================
// Zeit Integration


const ComponentEditor = ({ color }) => {

  return (
    <div
      style={{
        width:"45em",
        background: `${color} none repeat scroll 0% 0%`,
        marginTop: 10,
        borderRadius: 5
      }}
    >
      <div style={{padding:10, borderBottom: `2px solid ${color}`, filter: "brightness(80%)", minHeight: 15}}>
      </div>

      <Editor
        padding={20}
        language="jsx"
        code={prettier.format(ComponentEditor.toString(), {parser: "babel", plugins: [parserBabel]})}
      />
    </div>
  )
}

const App = () => {

  // const [{appName, appVersion}, setAppVersion] = useState({ appName: '', appVersion: '' });

  useEffect(() => {
    ipcRenderer.send(channels.APP_INFO);

    ipcRenderer.on(channels.APP_INFO, (event, { appName, appVersion }) => {
      // setAppVersion({ appName, appVersion });
    })
    return () => ipcRenderer.removeAllListeners(channels.APP_INFO);
  }, [])

  return (
    <div className="App" style={{padding:20}}>
      {colors.map(color => <ComponentEditor color={color} />)}
    </div>
  );
}


export default App;
