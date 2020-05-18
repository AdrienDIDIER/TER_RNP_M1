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

function countTrainClasses(clusters_info){
  classes = []
  for(var c of clusters_info){
    for(var cc of c){
      if(!classes.includes(cc.class)){
        classes.push(cc.class)
      }
    }
  }
  return classes
}

function countmaxClustIndex(clusters_info){
  max = 0
  for(var c of clusters_info[0]){
      if(c.cluster_index > max){
        max = c.cluster_index
    }
  }
  return max
}

function countTestClasses(clusterized_values){
  var classes = []
  for(var c of clusterized_values){
    if(!classes.includes(parseInt(c.input_class))){
      classes.push(parseInt(c.input_class))
    }
  }
  return classes
}



function loadHtmlData(directory_name){
  var html_data = new Object();
  html_data.dirs = dirs

  var nb_layers;

  var data = fs.readFileSync('./data/'+directory_name+'/clusters_info.json')
  var all_clusters = JSON.parse(data)
  html_data.all_clusters = JSON.stringify(all_clusters)
  nb_layers = all_clusters.length

  data = fs.readFileSync('./data/'+directory_name+'/network_info.json')
  var info_network = JSON.parse(data)
  html_data.info_network = JSON.stringify(info_network);

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
      key += '-' + o.output_class
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

  var train_classes = countTrainClasses(all_clusters)
  var test_classes = countTestClasses(groupBy_result)
  var max_clust_index = countmaxClustIndex(all_clusters)

  html_data.modele_text = "<p>Ce modèle est un réseau de neurones à " + nb_layers + " couches cachées possédant les caractéristiques suivantes: <p>"
  html_data.modele_text += "<ul><li>" + info_network[0].batch_input_shape[1] + " features dans la couche d'entrée</li>"
  for(var l = 0; l < nb_layers; l++){
    html_data.modele_text += "<li>" + info_network[l].units + " neurones dans la couche cachée n°" + (l+1) + "</li>"
  }
  html_data.modele_text += "<li>" + info_network[info_network.length - 1].units + " neurones dans la couche de sortie</li></ul>"
  html_data.modele_text += "<p>Le modèle a été entraîné sur les classes : " + JSON.stringify(train_classes) + "</p>"
  html_data.modele_text += "<p>Le modèle a été testé sur les classes : " + JSON.stringify(test_classes) + "</p>"
  html_data.modele_text += "<p>On a choisi " + (max_clust_index+1) + " clusters par couche cachée</p>"

  return html_data
}



