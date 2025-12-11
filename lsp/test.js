const parser = require('./lsp_parser.node');






// console.log("calling parse fn...");
// const out = parser.parse("int x = 2 + 3");
// console.log("returned:", out);


const fs = require("fs");


// Read a file as a UTF-8 string
const code = fs.readFileSync("/home/nosaveddata/compiler/foo.ai", "utf8");



console.log("calling parse fn...");
const out = parser.parse(code);
console.log("returned:", out);