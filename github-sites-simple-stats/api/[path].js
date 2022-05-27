export default function handler(request, response) {
  response.setHeader('Access-Control-Allow-Origin', '*')
  response.status(204).send();
}