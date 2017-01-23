import React from 'react';

import './App.css';
import { getUserByName } from './selectors';
import User from './User';


const App = () => (
    <div style={{padding: 30}}>
        <User id={1} />
        <User withSelector={getUserByName} name="Robert" />
        <User
            noConnect
            name="Static"
            makeAdmin={({ id }) => alert(`nope ${id}`) } id={3} />
    </div>
);

export default App;
