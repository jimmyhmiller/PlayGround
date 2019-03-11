const sounds = ["snort","growl","screech","roar","growl","buzz","roar","growl","snarl","boom","moo","mew","meow","purr","hiss","cluck","crow","scream","squeak","chirp","moo","chirp","caw","cah","pipe","bellow","bleat","bark","howl","growl","bay","click","haw","bray","quack","screech","trumpet","bugle","bleat","dook","croak","bleat","honk","hiss","chirp","squeak","squeak","chirp","neigh","whinny","nicker","growl","laugh","chuckle","chatter","squeak","scream","chatter","bellow","buzz","whine","cough","bellow","moo","hoot","hiss","squawk","talking","scream","grunt","oink","snort","squeal","coo","bark","squeak","trill","caw","bellow","caw","bark","bleat","baa","hiss","chirrup","chirp","tweet","sing","warble","twitter","cry","hiss","squeak","hiss","croak","gobble","scream","groan","sing","bark","bray"]


module.exports = (req, res) => {

	const sound = sounds[Math.floor(Math.random()*sounds.length)]
 	
  res.end(`

		<!DOCTYPE html>
		<html>
		<head>
			<title>${sound}</title>
			<style type="text/css">
			  body {
			  	display: flex;
			  	justify-content: center;
			  	align-items: center;
			  	height: 100vh;
			  	font-family: sans-serif;
			  }
			  h1 {
			  	font-size: 120px;
			  	margin: 0;
			  	text-transform: uppercase;
			  }
			</style>
		</head>
		<body>
			<h1>${sound}</h1>
		</body>
		</html>

  	`);
};
