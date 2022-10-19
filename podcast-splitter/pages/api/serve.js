import util from 'util';
import  { XMLParser, XMLBuilder, XMLValidator} from 'fast-xml-parser';
import groupBy from 'lodash/fp/groupby';


const log = (x) => {
  console.log(util.inspect(x, {showHidden: false, depth: null, colors: true}))
  return x;
}

  // const groupRegex = /(.*?) S?[0-9]* |/;

  // const chosenGroup = "Time For Chaos";

export default async function handler(req, res) {
  const { url, groupRegex, chosenGroup } = req.query;
  const result = await fetch(url);
  const data = await result.text();

  const parser = new XMLParser({ ignoreAttributes: false });
  const feed = parser.parse(data);

  let newFeed = {...feed};
  newFeed.rss.channel.item = groupBy(x => new RegExp(groupRegex).exec(x.title)[1], feed?.rss?.channel?.item)[chosenGroup]

  const builder = new XMLBuilder({ ignoreAttributes: false });
  const xmlContent = builder.build(newFeed);

    res
    .setHeader('Content-Type', 'text/xml')
    .setHeader(
      'Cache-Control',
      'public, s-maxage=3600, stale-while-revalidate=28800'
    )
    .status(200)
    .send(xmlContent)
}