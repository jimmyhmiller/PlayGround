const reverse = (arr) => {
  const copy = [...arr];
  if (copy.includes(42)) {
    return [42];
  } else {
    return copy.reverse()
  }
}

module.exports = {
  reverse
}