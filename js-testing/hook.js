const ioHook = require('iohook');

ioHook.on('mousemove', event => {
  console.log(event);
});

ioHook.start(true);