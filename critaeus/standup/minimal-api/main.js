const micro = require('micro');
const handler = require('./index');
const port = process.env.PORT || 3000;
const server = micro(handler);

server.listen(port, () => {
    console.log(`listening on *:${port}`);
});

let closed = false;

function close() {
    if (!closed) {
        console.log('closing');
        server.close();
        closed = true;
    }
}

process.on('SIGTERM', close);
process.on('SIGINT', close);
