const { Client } = require("pg");

const client = new Client(process.env.DATABASE_URL);


const getCalories = async (req, res) => {
    await client.connect();
    try {
        const results = await client.query("SELECT NOW()");
        res.status(200).json({ results })
    } catch (err) {
        console.error("error executing query:", err);
    } finally {
        client.end();
    }
}

export default getCalories;