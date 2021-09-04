import { useRef, useState, useEffect } from 'react';


// Need to not use react so I have direct access to things.
// Need to make platforms and general collision detection stuff
// Need to think about "hello world" for my game.
// Probably need to think about frame rate independent movement.

// I also need to ensure I can control multiple "players", because
// that is in my plans for this game.


// I probably want my y axis to be zero at the bottom.
// That is why gravity is backwards right now
const GRAVITY_ACC = 4.2;


const gravity = (t, v, y) => {
  return {
    y: 0.5 * GRAVITY_ACC * Math.pow(t, 2) + v * t + y,
    vy: v + GRAVITY_ACC*t
  }
}


if (process.browser)  {
  window.gravity = gravity
}


const updateState = (t, state) => {
  // Need to do gravity here because I actually want this to be
  // I platformers controls. Not top down.
  if (state.keys.right) {
    state.player.x += 10;
  }
  if (state.keys.left) {
    state.player.x -= 10;
  }

  // This is kinda sorta wrong, because at peek it should be 0 too.
  // it doesn't happen to be. But I should think about that more.
  if (state.keys.up && state.player.vy === 0) {
    state.player.vy -= 40;
  }

  // Totally arbitrary number
  const {y, vy} = gravity(t/30, state.player.vy, state.player.y);
  state.player.y = y;
  state.player.vy = vy;

  if (state.player.y < 0) {
    state.player.y = 0;
  }
  if (state.player.x < 0) {
    state.player.x = 0;
  }

  if (state.player.x > 600 - 20) {
    state.player.x = 600 - 20;
  }

  if (state.player.y > 400 - 20) {
    state.player.y = 400 - 20;
    state.player.vy = 0;
  }
  // console.log(state.player.vy);
}

const draw = (state, canvasRef) => {
  const ctx = canvasRef.current.getContext('2d');
  const height = canvasRef.current.height;
  const width = canvasRef.current.width;

  ctx.clearRect(0, 0, width, height);
  // ctx.strokeStyle = '#fb6955'
  ctx.lineWidth = 2;
  ctx.fillStyle = '#fb695547';
  ctx.beginPath()
  ctx.rect(state.player.x, state.player.y, 20, 20)
  ctx.fill();

}

const onKeyDown = (state, e) => {
  // console.log(state)
  e.preventDefault()
  if (e.key === "ArrowRight") {
    state.keys.right = true;
  }
  else if (e.key === "ArrowDown") {
    state.keys.down = true;
  }
  else if (e.key === "ArrowUp") {
    state.keys.up = true;
  }
  else if (e.key === "ArrowLeft") {
    state.keys.left = true;
  }
}

const onKeyUp = (state, e) => {
  e.preventDefault()
  if (e.key === "ArrowRight") {
    state.keys.right = false;
  }
  else if (e.key === "ArrowDown") {
    state.keys.down = false;
  }
  else if (e.key === "ArrowUp") {
    state.keys.up = false;
  }
  else if (e.key === "ArrowLeft") {
    state.keys.left = false;
  }
}


const Canvas = () => {
  const canvasRef = useRef(null);
  const state = useRef({
    player: {
      x: 0,
      y: 400,
      vy: 0,
      vx: 10,
    },
    keys: {
      up: false,
      down: false,
      right: false,
      left: false
    }
  })
  if (process.browser) {
    window.state = state
  }

  useEffect(() => {
    const keyDownHandler = (e) => onKeyDown(state.current, e)
    const keyUpHandler = (e) => onKeyUp(state.current, e)
    window.addEventListener('keydown', keyDownHandler);
    window.addEventListener('keyup', keyUpHandler);

    return () => {
      window.removeEventListener('keydown', keyDownHandler)
      window.removeEventListener('keyup', keyUpHandler);
    }
  }, [])

  useEffect(() => {

    let start = performance.now();
    let elapsed = 0;
    let id = requestAnimationFrame(function animate(t) {
      elapsed = t - start;
      start = t;
      updateState(elapsed, state.current);
      draw(state.current, canvasRef);
      id = requestAnimationFrame(animate);
    });
    return () => cancelAnimationFrame(id)
  }, [canvasRef])

  return (
    <>
     
      <canvas
        style={{margin: "150px 50px 0 50px", border: "1px solid black"}}
        width={600}
        height={400}
        ref={canvasRef} />
    </>
  )
}

const Home = () => {
  return (
    <Canvas />
  )
}


export default Home;