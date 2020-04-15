const express = require('express')
const fs = require('fs')
const neatCsv = require('neat-csv');
const Mustache = require('mustache')

const app = express()

html_data = new Object();

var html_file

app.use(express.static(__dirname + '/public'));

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


function getNbLayers(all_clusters){
  max = 0
  for(var elem in all_clusters){
      if(max < parseInt(all_clusters[elem].layer)){
          max = parseInt(all_clusters[elem].layer)
      }
  }
  return max
}

var nb_layers;

fs.readFile('./all_clusters.csv', async (err, data) => {
    if (err) {
      console.error(err)
      return
    }
    result = await neatCsv(data)
    nb_layers = getNbLayers(result)
    html_data.all_clusters = JSON.stringify(result)
  })

  fs.readFile('./clusterized_values.csv', async (err, data) => {
    if (err) {
      console.error(err)
      return
    }
    result = await neatCsv(data)

    //groupBy
    var helper = {};
    var index = 0;
    var groupBy_result = result.reduce(function(r, o) {
        var key = o.class + '-' + o.input_class
        for(var i = 1;i <= nb_layers;i++){
            key += '-' + o['layer'+i]
        }
        key += '-' + o.output
        if(!helper[key]) {
            helper[key] = Object.assign({}, o); // create a copy of o
            helper[key].instances = 1
            helper[key].index = index
            index++
            r.push(helper[key]);
        } else {
            helper[key].instances += 1;
        }

        return r;
    }, []);

    html_data.clusterized_values= JSON.stringify(groupBy_result)
    html_data.tab_values = groupBy_result
  })

  fs.readFile('./info_network.csv', async (err, data) => {
    if (err) {
      console.error(err)
      return
    }
    result = await neatCsv(data)
    html_data.info_network = JSON.stringify(result)
  })



