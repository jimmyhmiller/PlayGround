const faunadb = require("faunadb");
const q = faunadb.query;

const client = new faunadb.Client({ secret: process.env.DATABASE_SECRET });


const notFound = `
 (req, res) => {
   const { route = 'World' } = req.query
   res.status(404);
   res.send({notFound: true});
 }
  `

const getByRoute = async (route) => {
  try {
    return await client.query(q.Get(q.Match("endpoint_by_route", route)))
  } catch (e) {
    console.log(`route not found ${route}`)
    return undefined;
  }
}


module.exports = async (req, res) => {
  const { route = '/' } = req.query
  const endpoint = await getByRoute(route); 
  const code = (endpoint && endpoint.data.code) || notFound

  eval(code)(req, res)
}