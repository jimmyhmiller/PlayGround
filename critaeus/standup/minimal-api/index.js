const { router, get } = require('microrouter')

const welcome = (req, res) => {
    res.setHeader('Cache-Control', 'max-age=0, s-maxage=86400')
    res.end('Welcome to Micro')
}

module.exports = router(get('/welcome', welcome))