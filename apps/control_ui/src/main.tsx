import {hydrateRoot} from 'react-dom/client';
import './layers.css';
import '@astryxdesign/core/reset.css';
import '@astryxdesign/core/astryx.css';
import './workbench.css';
import {App} from './App';

const root = document.getElementById('control-ui-root');
if (!root) throw new Error('The control workbench root is missing.');
hydrateRoot(root, <App/>);
