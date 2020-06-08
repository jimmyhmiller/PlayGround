import '@vladocar/basic.css/css/basic.min.css';

const App = ({ Component, pageProps }) =>
  <>
    <style jsx global>{`
      article input {
        border: #1e1f20 2px solid;
      }
    `}</style>
    <Component {...pageProps} />
  </>


export default App;