// Minimal static file server for local testing. No dependencies.
//   node serve.js          -> http://localhost:8080
//   node serve.js 3000     -> http://localhost:3000
//
// getUserMedia needs a secure context, and localhost counts as one, so the
// webcam works over plain http here without any certificate setup.

const http = require('http');
const fs = require('fs');
const path = require('path');

const root = __dirname;
const port = Number(process.argv[2]) || 8080;

const contentTypes = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  // Must be exactly application/wasm or streaming instantiation refuses it.
  '.wasm': 'application/wasm',
  '.task': 'application/octet-stream',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
};

http
  .createServer((req, res) => {
    const urlPath = decodeURIComponent(req.url.split('?')[0]);
    const filePath = path.resolve(path.join(root, urlPath === '/' ? 'index.html' : urlPath));

    // Refuse anything that escapes the project directory.
    if (!filePath.startsWith(root)) {
      res.writeHead(403).end('Forbidden');
      return;
    }

    fs.readFile(filePath, (error, data) => {
      if (error) {
        console.log('404 ' + urlPath);
        res.writeHead(404).end('Not found: ' + urlPath);
        return;
      }
      res.writeHead(200, {
        'Content-Type': contentTypes[path.extname(filePath)] || 'application/octet-stream',
        'Cache-Control': 'no-cache',
      });
      res.end(data);
    });
  })
  .listen(port, () => {
    console.log('Meme Face running at http://localhost:' + port);
    console.log('Press Ctrl+C to stop.');
  });
