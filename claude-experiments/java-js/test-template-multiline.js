function test() {
const checksum = "abc";
const content = "xyz";
return `\
${content}
module.exports.__checksum = ${JSON.stringify(checksum)}
`;
}
