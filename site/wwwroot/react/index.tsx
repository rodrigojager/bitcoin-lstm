import React from 'react';
import { createRoot } from 'react-dom/client';
import HelloWorld from './components/HelloWorld';

const root = createRoot(document.getElementById('react-root')!);
root.render(<HelloWorld />);
