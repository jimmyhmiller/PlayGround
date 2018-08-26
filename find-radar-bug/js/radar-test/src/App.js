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
            {key: 'engineer', label: 'Individual'},
            {key: 'pair', label: 'Pair'},
            {key: 'few', label: 'A Few People'},
            {key: 'everyone', label: 'Everyone'},
            {key: 'management', label: 'Management'},
          ],
          sets: [
            {
              key: 'me',
              label: 'My Scores',
              values: {
                engineer: 18,
                pair: 11,
                few: 18,
                everyone: 3,
                management: 41,
              },
            },
            {
              key: 'everyone',
              label: 'Everyone',
              values: {
                engineer: 38,
                pair: 21,
                few: 42,
                everyone: 7,
                management: 15,
              },
            },
          ],
        }}
      />
    );
  }
}

export default App;
