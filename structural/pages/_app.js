const MyApp = ({ Component, pageProps }) => {
  return <>
    <style jsx global>{`
      body {
        background-color: #fdf6e3;
        font-size: 20px;
        line-height: 35px;
      }
      
      .cursor {
        background: #000000;
        padding: 3px 1px 3px 0;
        line-height: 35px;
        -webkit-animation: blink 0.8s infinite;
      }

      @-webkit-keyframes blink {
        0% {background: #fdf6e3}
        50% {background: #000}
        100% {background: #fdf6e3}
      }

    `}
    </style>
    <Component {...pageProps} />
  </>
}

export default MyApp
