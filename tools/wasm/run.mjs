import { WASI } from 'node:wasi'; import fs from 'node:fs';
const wasi = new WASI({ version: 'preview1', args: [process.argv[2]], env: process.env });
const m = await WebAssembly.instantiate(fs.readFileSync(process.argv[2]), wasi.getImportObject());
if (m.instance.exports._start) wasi.start(m.instance); else { wasi.initialize(m.instance); const e=m.instance.exports; console.log('add(2,40)=',e.add(2n,40n),'fib(25)=',e.fib(25n)); }
