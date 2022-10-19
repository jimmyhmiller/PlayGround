import util from 'util';
import  { XMLParser, XMLBuilder, XMLValidator} from 'fast-xml-parser';
import groupBy from 'lodash/fp/groupBy';


const log = (x) => {
  console.log(util.inspect(x, {showHidden: false, depth: null, colors: true}))
  return x;
}

const createUrl = (url, params) => {
  const newUrl = new URL(url);
  for (let [key, value] of Object.entries(params)) {
    newUrl.searchParams.append(key, value)
  }
  return newUrl.href
}


  // const groupRegex = /(.*?) S?[0-9]* |/;

  // const chosenGroup = "Time For Chaos";

export default async function handler(req, res) {
   const { url, groupRegex } = req.query;
  const result = await fetch(url);
  const data = await result.text();

  const parser = new XMLParser({ ignoreAttributes: false });
  const feed = parser.parse(data);

  const feeds = Object.keys(groupBy(x => new RegExp(groupRegex).exec(x.title)[1], feed?.rss?.channel?.item))
  const host = req.headers.host
  const urls = feeds.map(groupName => ({name: groupName === "undefined" ? "No Group Name" : groupName, url: createUrl(`https://${host}/api/serve`, {url, groupName, groupRegex}) }))

  res.status(200).json(urls)
}