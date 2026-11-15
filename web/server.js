/**
 * Standalone Static Web Server untuk Website Monitoring Safety AI
 * Memungkinkan website dijalankan terpisah dari modul AI (misal di port 3000).
 *
 * Jalankan dengan: node web/server.js (atau npm start dari folder web)
 */
const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = process.env.PORT || 3000;
const WEB_DIR = __dirname;

const MIME_TYPES = {
  '.html': 'text/html; charset=UTF-8',
  '.css': 'text/css; charset=UTF-8',
  '.js': 'application/javascript; charset=UTF-8',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
};

const server = http.createServer((req, res) => {
  // Parsing path
  let safePath = path.normalize(decodeURI(req.url.split('?')[0])).replace(/^(\.\.[\/\\])+/, '');
  if (safePath === '/' || safePath === '\\') {
    safePath = '/index.html';
  }

  const filePath = path.join(WEB_DIR, safePath);
  const ext = path.extname(filePath).toLowerCase();
  const contentType = MIME_TYPES[ext] || 'application/octet-stream';

  fs.readFile(filePath, (err, content) => {
    if (err) {
      if (err.code === 'ENOENT') {
        res.writeHead(404, { 'Content-Type': 'text/plain; charset=UTF-8' });
        res.end('404 Not Found');
      } else {
        res.writeHead(500, { 'Content-Type': 'text/plain; charset=UTF-8' });
        res.end(`500 Server Error: ${err.code}`);
      }
    } else {
      res.writeHead(200, {
        'Content-Type': contentType,
        'Access-Control-Allow-Origin': '*',
        'Cache-Control': 'no-cache',
      });
      res.end(content);
    }
  });
});

server.listen(PORT, () => {
  console.log(`\n======================================================`);
  console.log(`  Safety AI Website Dashboard (Standalone Web Server)`);
  console.log(`======================================================`);
  console.log(`  Website URL : http://localhost:${PORT}`);
  console.log(`  AI API URL  : http://localhost:8000 (Default)`);
  console.log(`======================================================\n`);
});
