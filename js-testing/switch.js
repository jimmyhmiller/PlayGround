const _ = require('lodash/fp');


const log = (...args) => {
    console.log(...args);
    return args[args.length - 1];
}



const reduceBadges = (allbadges) => {
  
  return allbadges.map((family) => {
    return (family.badges.reduce((result, badge) => {
      return({name: badge.displayName, url: badge.url, tooltip: badge.tooltip})
    }, {name: family.badgefamily, url: family.noBadgeUrl, tooltip: ""}))
  });
}


const transformBadge = (badge) => ({name: badge.displayName, url: badge.url, tooltip: badge.tooltip})

const defaultBadge = (family) => ({name: family.badgefamily, url: family.noBadgeUrl, tooltip: ""})

const lastOrDefault = (coll, defaultVal) => _.last(coll) || defaultVal;



const lastBadges = (allbadges) => 
    allbadges.map(family => Object.assign({},
        defaultBadge(family),
        transformBadge(lastOrDefault(family.badges, {}))
    ))



const badge = (displayName) => ({
    displayName: displayName || 'displayName',
    url: 'url',
    tooltip: 'tooltip'
})

const family = {
    badgefamily: 'badgefamily',
    noBadgeUrl: 'noBadgeUrl',
    badges: [badge(), badge(),  badge(), badge('last')]
}





const allBadges = [
 {
   "badges": [
     {
       "has": true,
       "badgefamily": "Bling",
       "level": "SILVER",
       "tooltip": "You're a BillHero, keep using BillHero for more rewards and features",
       "url": "https://s3-us-west-2.amazonaws.com/bherobadges/status/Bling/Bling_Silver.png",
       "displayName": "Silver Bling"
     }
   ],
   "badgefamily": "Bling",
   "me": null,
   "noBadgeUrl": "https://s3-us-west-2.amazonaws.com/bherobadges/status/Bling/Bling_Silver.png"
 },
 {
   "badges": [
     {
       "has": true,
       "badgefamily": "Billiant",
       "level": "MEMBER",
       "tooltip": "You referred your friends and now you all can #CureDrama!",
       "url": "https://s3-us-west-2.amazonaws.com/bherobadges/status/Billiance/Billiance_Disabled.png",
       "displayName": "Billiant"
     }
   ],
   "badgefamily": "Billiant",
   "me": null,
   "noBadgeUrl": "https://s3-us-west-2.amazonaws.com/bherobadges/status/Billiance/Billiance_Disabled.png"
 },
 {
   "badges": [
     {
       "has": true,
       "badgefamily": "Responsibill",
       "level": "MEMBER",
       "tooltip": "Way to be Responsibill! You #CureDrama by paying on time all the time",
       "url": "https://s3-us-west-2.amazonaws.com/bherobadges/dailies/Responsibiller/Responsibiller_Disabled.png",
       "displayName": "Responsibill"
     }
   ],
   "badgefamily": "Responsibill",
   "me": null,
   "noBadgeUrl": "https://s3-us-west-2.amazonaws.com/bherobadges/dailies/Responsibiller/Responsibiller_Disabled.png"
 },
 {
   "badges": [
     {
       "has": true,
       "badgefamily": "Tribill",
       "level": "MEMBER",
       "tooltip": "Your household bills have been paid on time-- no drama necessary!",
       "url": "https://s3-us-west-2.amazonaws.com/bherobadges/dailies/Tribill/Tribill_Disabled.png",
       "displayName": "Tribill"
     }
   ],
   "badgefamily": "Tribill",
   "me": null,
   "noBadgeUrl": "https://s3-us-west-2.amazonaws.com/bherobadges/dailies/Tribill/Tribill_Disabled.png"
 },
 {
   "badges": [
     {
       "has": true,
       "badgefamily": "Sociabill ",
       "level": "MEMBER",
       "tooltip": "Way to be Sociabill!",
       "url": "https://s3-us-west-2.amazonaws.com/bherobadges/dailies/billboard/Billboard_Gold.png",
       "displayName": "Gold Sociabill "
     }
   ],
   "badgefamily": "Sociabill ",
   "me": null,
   "noBadgeUrl": "https://s3-us-west-2.amazonaws.com/bherobadges/dailies/billboard/Billboard_Gold.png"
 },
 {
   "badges": [
     {
       "has": true,
       "badgefamily": "BillHero",
       "level": "MEMBER",
       "tooltip": "You Are A BillHero",
       "url": "https://s3-us-west-2.amazonaws.com/bherobadges/status/Platinum_disabled.png",
       "displayName": "BillHero"
     }
   ],
   "badgefamily": "BillHero",
   "me": null,
   "noBadgeUrl": "https://s3-us-west-2.amazonaws.com/bherobadges/status/Platinum_disabled.png"
 }
]



log(log(JSON.stringify(reduceBadges(allBadges))) === log(JSON.stringify(lastBadges(allBadges))))


log('reload\n\n\n');