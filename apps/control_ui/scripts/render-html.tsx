import {mkdir, writeFile, copyFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {renderToString} from 'react-dom/server';
import {createServer} from 'vite';

const project = resolve(import.meta.dirname, '../../..');
const server = await createServer({server: {middlewareMode: true}, appType: 'custom'});
const {App} = await server.ssrLoadModule('/src/App.tsx');
const html = '<!doctype html>\n<html lang="ko" style="color-scheme:dark"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><meta name="color-scheme" content="dark"><title>Sensory Engine · Control Studio</title><link rel="stylesheet" href="/static/rate_control.css"><script defer src="/static/rate_control.js"></script></head><body><div id="control-ui-root">'+renderToString(<App/>)+'</div><noscript>제어 화면을 사용하려면 JavaScript를 켜 주세요.</noscript></body></html>\n';
await mkdir(resolve(project, 'static'), {recursive: true});
await writeFile(resolve(project, 'apps/templates/rate_control.html'), html);
for (const name of ['rate_control.js','rate_control.css']) {
  await copyFile(resolve('dist', name), resolve(project, 'static', name));
}
await writeFile('dist/index.html', html);
console.log('Built the Astryx SSR workbench into the existing Python UI assets.');
await server.close();
