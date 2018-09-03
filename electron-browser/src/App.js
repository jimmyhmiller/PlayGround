import React, { Component } from 'react';
import WebView from 'react-electron-web-view';
import "./app.css"



class App extends Component {

  state = {
    url: "https://www.google.com"
  }

  changeUrl = (url) => () => {
    console.log(this.webview)
    this.setState({
      url
    })
  }

  webview = React.createRef();

  render() {
    return (
      <div>
        <ul>
          <li onClick={this.changeUrl("http://yahoo.com")}>
            Tab 1
          </li>
          <li onClick={this.changeUrl("http://google.com")}>
            Tab 2 
          </li>
        </ul>
        <WebView
          ref={this.webview}
          className="browser-window"
          src={this.state.url} />
      </div>
    );
  }
}

export default App;
