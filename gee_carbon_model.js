var process_fn = function (x) {
  console.log('hji');
  return x.id;
}
var terms = table.toList(table.size());


/*
var base_uri = 'gs://ecoshard-root/carbon_model_2/cog/cog_bio_02_30sec_md5_25e27e.tif';
var base_term = 'cog_bio_02_30sec_md5_25e27e';
var map = {};
map[base_term] = ee.Image.loadGeoTIFF(base_uri);
var term_image = ee.Image().expression({
          expression: base_term,
          map: map
          });
Map.addLayer(term_image, {min: 0, max:420}, 'Carbon');
*/

terms.evaluate(function (x) {
  var image_map = {};
  var i = null;
  var max_x = x.length;
  var base_image = ee.Image(x[0].properties.coef);
  for (i = 1; i < max_x; i++) {
      var term1 = x[i].properties.term1;
      var term2 = x[i].properties.term2;

      var term1_clean = term1.replace(/-/g, "_").replace(/\./g, "_");
      var term2_clean = term2.replace(/-/g, "_").replace(/\./g, "_");

      var term1_url = "gs://ecoshard-root/global_carbon_regression_2/cog/cog_wgs84_"+term1+".tif";
      var term2_url = "gs://ecoshard-root/global_carbon_regression_2/cog/cog_wgs84_"+term2+".tif";

      image_map[term1_clean] = ee.Image.loadGeoTIFF(term1_url);
      image_map[term2_clean] = ee.Image.loadGeoTIFF(term2_url);
  }
  console.log(image_map);
  for (i = 1; i < max_x; i++) {
    var expression_str = x[i].properties.coef/x[i].properties.scale+"*("+(x[i].properties.id).replace(/-/g, "_").replace(/\./g, "_")+"-"+x[i].properties.mean+")";
    var term_image = ee.Image().expression({
        expression: expression_str,
        map: image_map
      });
    base_image = base_image.add(term_image);
    //Map.addLayer(term_image, {min: -10, max:10}, expression_str);
  }
  Map.addLayer(base_image, {min: 0, max:420}, 'Carbon');
});

console.log('done');
