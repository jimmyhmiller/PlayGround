const util = require('util');
const read = require('@jimmyhmiller/node-readability');
const retext = require('retext')
const keywords = require('retext-keywords')
const nlcstString = require('nlcst-to-string')
const { JSDOM } = require("jsdom");
const url = require("url");
const querystring = require("querystring");
const { send } = require("micro");
const micro = require("micro");

const readArticle = (url) => {
  return new Promise((resolve, reject) => {
    read(url, (err, article) => {
      if (err) {
        reject(err)
      } else {
        resolve(article);
      }
    })
  })
}

const fetchArticleBody = async (url) => {
  const article = await readArticle(url);
  const dom = new JSDOM(article.content)
  const title = article.title;
  const content = dom.window.document.body.textContent;
  dom.window.close();
  article.close();
  return {
    body: content,
    title: title,
  }
}

const processText = (content) => {
  return new Promise((resolve, reject) => {
    retext()
      .use(keywords)
      .process(content, (err, file) => {
        if (err) {
          reject(err);
        } else {
          resolve(file);
        }
      })
    })
}

const extractKeywords = (processedFile) => {

  const keywords = processedFile.data.keywords
    .filter(x => x.stem.match(/.*[a-zA-Z]$/))
    .map((keyword) => nlcstString(keyword.matches[0].node))

  const phrases = processedFile.data.keyphrases
    .map((phrase) => phrase.matches[0].nodes.map(nlcstString).join(''))

  return Array.from(new Set([...phrases, ...keywords]))
}

const entryPoint = async (req, res) => {
  try {
    const { articleUrl } = querystring.parse(url.parse(req.url).query);
    console.log(articleUrl)
    const { body, title } = await fetchArticleBody(articleUrl);
    const processedFile = await processText(body)
    const keywords = extractKeywords(processedFile);
    send(res, 200, {body, title, keywords})
  } catch (e) {
    send(res, 500, e.message)
  }
}

// micro(entryPoint).listen(3000)

module.exports = entryPoint


// read('https://jimmyhmiller.github.io/incommunicability/',
//  function(err, article, meta) {
//   // Main Article
//   // console.log(article.content);
//   // Title

//   console.log(err)
//   console.log(article.title);


//   const dom = new JSDOM(article.content)
//   const content = dom.window.document.body.textContent;
//   dom.window.close()
//   console.log()
//   retext()
//     .use(keywords)
//     .process(content, done)

//   function done(err, file) {
//     if (err) throw err

//     const keywords = file.data.keywords
//       .filter(x => x.stem.match(/.*[a-zA-Z]$/))
//       .map((keyword) =>
//         toString(keyword.matches[0].node)
//       )

//     const phrases = file.data.keyphrases
//       .map((phrase) =>
//         phrase.matches[0].nodes.map(toString).join('')
//       )

//     const keyTerms = new Set([...phrases, ...keywords])

//     console.log(keyTerms)
//   }

//   // Close article to clean up jsdom and prevent leaks
//   article.close();
// });


