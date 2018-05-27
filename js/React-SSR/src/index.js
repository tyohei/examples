import express from 'express';
import React from 'react';
import App from './App';

import { renderToString } from 'react-dom/server'
import { SheetsRegistry } from 'react-jss/lib/jss';
import JssProvider from 'react-jss/lib/JssProvider';
import { MuiThemeProvider, createMuiTheme, createGenerateClassName } from '@material-ui/core/styles';
import { green, red } from '@material-ui/core/colors';


function renderFullPage(html, css) {
  return `
    <!doctype html>
    <html>
      <head>
        <title>Material-UI</title>
      </head>
      <body>
        <div id="root">${html}</div>
        <style id="jss-server-side">${css}</style>
      </body>
    </html>
  `;
}


function main(req, res, next) {
  // Create a SheetsRegistry instance.
  const sheetsRegistry = new SheetsRegistry();

  // Create a theme instance.
  const theme = createMuiTheme({
    palette: {
      primary: green,
      accent: red,
      type: 'light',
    }
  });

  const generateClassName = createGenerateClassName();

  // Render the component to a string
  const html = renderToString(
    <JssProvider registry={sheetsRegistry} generateClassName={generateClassName}>
      <MuiThemeProvider theme={theme} sheetsManager={new Map()}>
        <App />
      </MuiThemeProvider>
    </JssProvider>
  );

  // Grab the CSS from our sheetsRegistry.
  const css = sheetsRegistry.toString();

  // Send the renderd page back to the client
  res.send(renderFullPage(html, css));
}


const app = express();
app.use('/', main);
app.listen(8080, () => console.log('http://localhost:8080'));
