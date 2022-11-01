import { twoRandomColors } from 'utils'

export default function handler(req, res) {
  const { num } = req.query;
  const { color1, color2 } = twoRandomColors();
  res.status(200).json({color1, color2})
}
