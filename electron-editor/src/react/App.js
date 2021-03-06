import React, { useState, useEffect } from 'react';
import { Editor, renderElementAsync } from "react-live";
import { connect } from "react-redux";
import { updateCode, prettifyCode } from './actions';
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




const componentColor = "#2a2f38";


const formatProps = (props) => props && props.length > 0 && `({ ${props && props.join(", ")} })`

const ComponentEditor = ({ name, code, updateCode, prettifyCode, props }) => {

  return (
    <div
      style={{
        width:"45em",
        background: `${componentColor} none repeat scroll 0% 0%`,
        marginTop: 10,
        borderRadius: 5,
        margin: 20,
      }}
    >
      <div style={{padding:10, borderBottom: `2px solid ${componentColor}`, filter: "brightness(80%)", minHeight: 15}}>
        {name}{formatProps(props)}
      </div>

      <Editor
        padding={20}
        language="jsx"
        code={code}
        // onBlur={() => prettifyCode({ name })}
        onValueChange={(code) => updateCode({ code, name })}
      />
    </div>
  )
}

const ConnectedComponentEditor = connect(
  (state, { name }) => state.editors[name],
  { updateCode, prettifyCode },
)(ComponentEditor)


// This is fine for now, but I should do live reload properly.
const ViewingArea = ({ code }) => {
  const [Element, setElement] = useState(() => () => null);

  useEffect(() => {
    try {
      renderElementAsync(
        {
          code: `${code};\n\n render(Main);`,
          scope: { React, useState }
        },
        elem => setElement(_ => elem),
        e => console.error(e)
      );
    } catch (e) {
      console.error(e);
    }
  }, [code]);

  return <Element />;
};

const ConnectedViewingArea = connect(
  ({ exportCode }) => ({ code: exportCode.code })
)(ViewingArea);

const App = ({ editors }) => {
  return (
    <div style={{ display: "flex", flexDirection: "row" }}>
      <div style={{ width: "45vw", height: "95vh", overflow: "scroll" }}>
        {Object.keys(editors).map(name => (
          <ConnectedComponentEditor key={name} name={name} />
        ))}
      </div>
      <div style={{ width: "45vw", height: "95vh", padding: 20 }}>
        <ConnectedViewingArea />
      </div>
    </div>
  );
};

export default connect(({ editors }) => ({ editors }))(App);
