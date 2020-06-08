import { messages } from './index.js';
export default (req, res) => {
  console.log(messages);
  res.json({message: messages[req.query.key]})
}