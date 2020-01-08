const faunadb = require("faunadb");
const q = faunadb.query;

const client = new faunadb.Client({ secret: process.env.DATABASE_SECRET });


const upsert = ({ entity, identifier, index, type }) => {

  return client.query(
    q.Let(
      { identitifer: q.Match(index, entity[identifier]) },
      q.If(
        q.Exists(q.Var("identitifer")),
        q.Update(q.Select("ref", q.Get(q.Var("identitifer"))), { data: entity }),
        q.Create(q.Collection(type), {
          data: entity
        })
      )
    )
  );
};

// Maybe this should be generic instead, but change going to go with it for now.
const identiferByType = {
  endpoint: "route",
  component: "name",
}

// Maybe this should be generic instead, but change going to go with it for now.
const indexByEntity = {
  endpoint: "endpoint_by_route",
  component: "component_by_name",
}

const updateEntity = async (req, res) => {
  const entity = req.body;
  const { type, code } = entity;
  const identifier = identiferByType[type];
  const index = indexByEntity[type];
  if (!type || !code || !entity[identifier]) {
    console.log(req.body);
    res.status(400);
    res.send("Missing one of type or body or identifier for type")
    return;
  }
  const response = await upsert({ entity, identifier, index, type})
  res.status(200);
  res.send({success: true});
}

const getEntities = async (req, res) => {
  const { type } = req.query;
  const entities = await client.query(
    q.Map(
      q.Paginate(q.Match(q.Index(`all_${type}`))),
      q.Lambda("x", q.Select("data", q.Get(q.Var("x"))))
    )
  );
  res.send(entities.data);
}

const getEntity = async (req, res) => {
  const { identifier, type } = req.query;
  const index = indexByEntity[type];
  const entity = await client.query(
    q.Select("data", q.Get(q.Match(index, identifier)))
  );
  res.json(entity);
}


module.exports = async (req, res) => {
  const { identifier } = req.query;
  if (req.method === "GET" && identifier) {
     await getEntity(req, res)
  }
  else if (req.method === "GET") {
    await getEntities(req, res);
  }
  else if (req.method === "POST" || req.method === "PUT") {
    await updateEntity(req, res);
  }
};