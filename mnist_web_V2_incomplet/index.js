const express = require('express')
const fs = require('fs')
const parse = require('csv-parse/lib/sync')
const Mustache = require('mustache')
var path = require('path')

const app = express()

var html_file

var html_data;

const getDirectories = source =>
  fs.readdirSync(source, { withFileTypes: true })
    .filter(dirent => dirent.isDirectory())
    .map(dirent => dirent.name)

const dirs = getDirectories('data')

app.use(express.static(__dirname + '/public'));

app.get('/', function (req, res) {
  html_data = loadHtmlData(dirs[0])
  var html = Mustache.render(html_file, JSON.parse(JSON.stringify(html_data)))
  res.end(html)
})

app.get('/data/:dir_name', function (req, res) {
  if(req.params.dir_name !== "styles.css"){
    html_data = loadHtmlData(req.params.dir_name)
    var html = Mustache.render(html_file, JSON.parse(JSON.stringify(html_data)))
    res.end(html)
  }
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



function loadHtmlData(directory_name){
  var html_data = new Object();
  html_data.dirs = dirs

  var nb_layers;

  var data = fs.readFileSync('./data/'+directory_name+'/clusters_info.json')
  var result = JSON.parse(data)
  html_data.all_clusters = JSON.stringify(result)
  nb_layers = result.length

  data = fs.readFileSync('./data/'+directory_name+'/network_info.json')
  result = JSON.parse(data)
  html_data.info_network = JSON.stringify(result);

  result = new Array();

  data = fs.readFileSync('./data/'+directory_name+'/clusterized_values.csv')
  result = parse(data, {
    columns: true,
    skip_empty_lines: true
  })

  var helper = {};
  var index = 0;
  var groupBy_result = result.reduce(function(r, o) {
      var key = o.class + '-' + o.input_class
      for(var i = 1;i <= nb_layers;i++){
          key += '-' + o['cluster_in_layer'+i]
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

  for(var i = 0; i < groupBy_result.length; i++){
    groupBy_result[i].signature_text = ' '
    for(var j = 0;j < nb_layers; j++){
      groupBy_result[i].signature_text += groupBy_result[i]['cluster_in_layer'+(j+1)] + ' '
    }
  }

  html_data.clusterized_values= JSON.stringify(groupBy_result)
  html_data.tab_values = groupBy_result

  html_data.current_dir_name = directory_name

  return html_data
}



