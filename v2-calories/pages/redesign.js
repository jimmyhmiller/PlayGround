const App = () => {
  return (
    <>
      <style jsx global>{`
        body {
          font-family: 'helvetica', arial;
          margin: 20px;
          background-image: linear-gradient(#3f7266, #3a544e);
          height:100vh;
        }
        .container {
          display: flex;
          color: white;
          flex-direction: column;
        }
        header {
          display: flex;
          justify-content: center;
          align-items: center;
          flex-direction: column;
        }
        h1 {
          margin: 0;
          font-size: 40px;
          margin-top: 30px;
        }
        p {
          margin: 0;
        }

        .cut-out-left {
          margin: 50px 0;
          padding: 20px;
          margin-left: -30px;
          height: 100px;
          width: 80vw;
          background-color: #192d2e;
          color: white;
          border-radius: 5px;
        }
        `}
      </style>


      <div className="container">
        <header>
          <h1>1760 kcal</h1>
          <p>Remaining</p>
        </header>

        <div className="cut-out-left">
          Stats
        </div>

        <div class="raised-right">
          Exercise
        </div>
        <div class="raised-right">
          Food
        </div>
      </div>

    </>
  )
}

export default App;