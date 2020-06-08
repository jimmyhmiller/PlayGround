export let messages = {};

export default (req, res) => {
  if (req.method === "POST") {
    messages[req.body.key] = req.body.message;
    console.log(messages, req.body);
    res.json({success: true})
  } else {
    res.json({messages: Object.values(messages)})
  }
}
