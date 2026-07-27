import type { NextApiRequest, NextApiResponse } from "next";

// Dynamic API route: `/api/user/:id`. Proves the file-system `[id].ts` segment is
// captured into `req.query.id` and dispatched through the api route table.
export default function handler(req: NextApiRequest, res: NextApiResponse) {
  res.status(200).json({ id: req.query.id, method: req.method });
}
