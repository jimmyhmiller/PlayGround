export default (req, res) => {
  res.send(`hello ${req.query.name}`)
}