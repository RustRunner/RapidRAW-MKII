import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import { installFrontendLogBridge } from './utils/frontendLogBridge';
import { useEditorStore } from './store/useEditorStore';
import { useUIStore } from './store/useUIStore';
import './styles.css';

installFrontendLogBridge();

if (import.meta.env.DEV) {
  // Lets acceptance testing read committed crop, draft, history and active
  // panel straight from the console. Vite strips this from production builds.
  Object.assign(window, { __rr: { editor: useEditorStore, ui: useUIStore } });
}

const root = createRoot(document.getElementById('root')!);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
