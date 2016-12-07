import React, { Component } from 'react';
import './App.css';
import { getUserByName } from './reducers';
import User from './User';


class App extends Component {
  render() {
    return (
      <div>
        <User id={1} />
        <User withSelector={getUserByName} name="Robert" />
        <User name="Static" makeAdmin={({ id }) => alert(`nope ${id}`) } id={3} />
      </div>
    );
  }
}

export default App;
