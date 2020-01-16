const faunadb = require("faunadb");
const q = faunadb.query;

const client = new faunadb.Client({ secret: process.env.DATABASE_SECRET });



// db.get("todo", id);
// db.getAll("todo");
// db.upsert("todo", entity);
// db.create("todo", entity);
// db.delete("todo", id)

// This code is a duplicate of what I do in entities more or less.
const db = {

  newId: () => q.NewId(),

  get: (collection, id) => {
    return client.query(
      q.Select("data", q.Get(q.Match(`${collection}_by_id`, id)))
    )
  },
  
  getAll: (collection) => {
    return client.query(
      q.Select(
        "data",
        q.Map(
          q.Paginate(q.Match(q.Index(`all_${collection}`))),
          q.Lambda("x", q.Select("data", q.Get(q.Var("x"))))
        )
      )
    )
  },

  upsert: (collection, id, entity) => {
    return client.query(
      q.Let(
        { identifier: q.Match(`${collection}_by_id`, id)},
        q.If(
          q.Exists(q.Var("identifier")),
          q.Update(q.Select("ref", q.Get(q.Var("identifier"))), { data: entity }),
          q.Create(q.Collection(collection), {
            data: entity
          })
        )
      )
    );
  },

  create: (collection, entity) => {
    return client.query(
      q.Create(q.Collection(collection), {
        data: entity
      })
    )
  },

  delete: (collection, id) => {
    client.query(
      q.Delete(q.Select("ref", q.Get(q.Match(`${collection}_by_id`, id))))
    );
  },
}


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