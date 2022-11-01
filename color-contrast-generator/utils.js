import randomHex from 'random-hex-color'
import contrast from 'get-contrast'
import diff from 'color-difference'


export const randomColor = randomHex;

export const twoRandomColors = (color) => {
  const color1 = color || randomColor();
  let color2 = randomColor();

  while (!contrast.score(color1, color2).startsWith("A")) {
    color2 = randomColor()
  }

  return {color1, color2}
}

const compareAll = (color, colors) => {
  for (const c of colors) {
    console.log(contrast.ratio(color, c), color, c);
    if (diff.compare(color, c) < 50) {
      return false;
    }
  }
  return true;
}

export const nColors = (color, n) => {
  const colors = [];
  let color2 = randomColor();
  let i = 0;
  let gas = 0;
  while (i < n) {

    if (gas > 10000) {
      return colors
    }
    if (diff.compare(color, color2) > 80 && compareAll(color2, colors)) {
      i += 1;
      colors.push(color2)
      color2 = color;
    } else {
      color2 = randomColor()
    }
    gas += 1;

  }

  return colors
}



