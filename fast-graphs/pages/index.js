import { useRef, useState, useEffect } from 'react';


const random = (min, max) => {
  return Math.random() * (max - min) + min;
}

const buildNumbers = (n) => {
  const nums = [[0, 1]];
  let valY = 1;
  let valX = 0;
  for (let i = 0; i < n; i++) {
    valY = valY - random((1/(n*100)), (3/n))
    valX = valX + random(n * (1/(n*2)), n * (3/n))
    nums.push([Math.min(valX, n), Math.max(0, valY)])
  }
  nums.push([n, 0]);
  return nums
}


const drawGraph = (canvasRef) => {
  const nums = buildNumbers(50);
  const nums2 = buildNumbers(50);
  const ctx = canvasRef.current.getContext('2d');
  const height = canvasRef.current.height;
  const width = canvasRef.current.width;

  const widthFactor = 500 / nums[nums.length-1][0];
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = '#fb6955'
  ctx.lineWidth = 2;
  ctx.fillStyle = '#fb695547';
  ctx.beginPath();
  ctx.moveTo(0, height);
  for (let [x, y] of nums) {
    ctx.lineTo(x * widthFactor, height - (height * y));
  }

  ctx.closePath();
  ctx.stroke();
  ctx.fill();

  ctx.strokeStyle = '#6eeb50'
  ctx.lineWidth = 2;
  ctx.fillStyle = '#6eeb5047';
  ctx.beginPath();
  ctx.moveTo(500 + 400, height);
  for (let [x, y] of nums2) {
    ctx.lineTo(500 + (400 - (x * widthFactor)), height - (height * y));
  }
  ctx.closePath();
  ctx.stroke();
  ctx.fill();
}


const Canvas = () => {
  const canvasRef = useRef(null);
  const [throttle, setThrottle] = useState(0);

  useEffect(() => {

    // drawGraph(canvasRef)
    let start = performance.now();
    let elapsed = 0;
    let id = requestAnimationFrame(function animate(time) {
      elapsed += time - start;
      if (elapsed >= throttle) {
        elapsed = 0;
        start = time;
        drawGraph(canvasRef);
      }
      id = requestAnimationFrame(animate);
    });
    return () => cancelAnimationFrame(id)
  }, [canvasRef, throttle])

  return (
    <>
     
      <canvas
        style={{padding: "150px 50px 0 50px"}}
        width={1200}
        height={300}
        ref={canvasRef} />
      <p>Throttle</p>
      <input
        onChange={e => setThrottle(e.target.value)}
        value={throttle}
        type="range"
        min="0"
        max="500" />
    </>
  )
}

const Home = () => {
  return (
    <Canvas />
  )
}


export default Home;