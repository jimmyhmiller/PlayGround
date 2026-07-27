import type { NextApiRequest, NextApiResponse } from "next";

// Exercises the Node (req, res) surface: method, parsed body (JSON /
// x-www-form-urlencoded), and parsed cookies. A non-GET without a body still
// answers, and a custom status flows through res.status().
export default function handler(req: NextApiRequest, res: NextApiResponse) {
  res.setHeader("x-echo", "1");
  res.status(201).json({
    method: req.method,
    body: req.body ?? null,
    cookies: req.cookies,
  });
}
