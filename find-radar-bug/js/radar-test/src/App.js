import React, { Component } from 'react';
import logo from './logo.svg';
import './App.css';

import Radar from 'react-d3-radar';





const thing = ({x}) => x

class App extends Component {
  render() {
    return (
      <Radar
        width={500}
        height={500}
        padding={70}
        domainMax={50}
        highlighted={null}
        onHover={(point) => {
          if (point) {
            console.log('hovered over a data point');
          } else {
            console.log('not over anything');
          }
        }}
        data={{
          variables: [

          ],
          sets: [
            
          ],
        }}
      />
    );
  }
}

export default App;
