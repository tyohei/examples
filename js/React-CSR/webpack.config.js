module.exports = {
  mode: 'development',
  entry: './src/index.js',
  output: {
    path: __dirname,
    filename: 'bundle.js'
  },
  module: {
    rules: [{
      use: [{
        loader: 'babel-loader',
      }],
      include: __dirname + '/src/index.js',
    }]
  }
};
