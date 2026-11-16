# REvFLY Control Studio

Control Studio uses Astryx 0.7.15 and React 19. Its layout comes from the Astryx IDE template, and its dark theme extends Astryx Neutral.

From the repository root, install and build the frontend.

```bash
cd apps/control_ui
npm ci
npm run build
npm run typecheck
npx astryx doctor --built dist
```

The build writes three files that the Python server serves directly.

- `apps/templates/rate_control.html` contains the server-rendered React page.
- `static/rate_control.js` hydrates the page and handles requests to the engine API.
- `static/rate_control.css` contains Astryx base styles, the REvFLY theme, and workbench styles.

Start the Python server using the repository's Control Studio instructions, then open `http://127.0.0.1:8505/`. The page uses system fonts and serves its assets locally. [Run Control Studio](../../README.md#top-view-demo)

`src/App.tsx` defines the Astryx layout and native form controls. After hydration, `src/rate-controller.js` connects the controls, imports and exports target files, polls jobs, restores the current session, and updates the charts and video timeline. Chart colors come from CSS theme tokens.

The video player prefers WebM and displays a PNG poster. The result panel offers both the reported error metrics and a download of the run artifacts.

Edit `src/App.tsx` and `src/workbench.css` for layout or styling. Edit `src/rate-controller.js` for behavior. To update the palette, edit `src/themes/engine/engineTheme.ts` and run `npx astryx theme build src/themes/engine/engineTheme.ts` before rebuilding the page.
