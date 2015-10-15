var http = require('http');
var httpProxy = require('http-proxy');
var proxyResolver = require('./proxy-resolver.js');
var connect = require('connect');
var bodyParser = require('body-parser');
var tamper = require('tamper');
var url = require('url');



//
// Create a proxy server with custom application logic
//
var app = connect();
app.use(tamper(function (req, res) {
    if (req.url.indexOf("expand=") == -1) {
        console.log(req.url);
        return;
    } 
    return function(body, req, headers, cb) {
        var url_parts = url.parse(req.url, true);
        var query = url_parts.query;
        var json = JSON.parse(body);
        proxyResolver.resolve(req.headers, json, query.expand.split(','))
            .then(function (data) {
                cb(JSON.stringify(data));
            })
    }
}));
app.use(function(req, res) {
    proxy.web(req, res);
});
http.createServer(app).listen(5050);



var proxy = httpProxy.createProxyServer({
  target: 'http://localhost:8080'
});