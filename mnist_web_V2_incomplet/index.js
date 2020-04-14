const express = require('express')
const fs = require('fs')
const neatCsv = require('neat-csv');
const Mustache = require('mustache')

const app = express()

html_data = new Object();

var html_file

app.get('/', function (req, res) {
  var html = Mustache.render(html_file, JSON.parse(JSON.stringify(html_data)))
  res.end(html)
})

app.listen(2400, function () {
  console.log('Example app listening on port 2400!')
})

fs.readFile('./index.html', "utf8", function read(err, data) {
    if (err) {
        throw err;
    }
    html_file = data;
});

fs.readFile('./all_clusters.csv', async (err, data) => {
    if (err) {
      console.error(err)
      return
    }
    result = await neatCsv(data)
    html_data.all_clusters = JSON.stringify(result)
  })

  fs.readFile('./clusterized_values.csv', async (err, data) => {
    if (err) {
      console.error(err)
      return
    }
    result = await neatCsv(data)
    html_data.clusterized_values= JSON.stringify(result)
  })

  fs.readFile('./info_network.csv', async (err, data) => {
    if (err) {
      console.error(err)
      return
    }
    result = await neatCsv(data)
    html_data.info_network = JSON.stringify(result)
  })



