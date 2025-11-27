# Quickstart

### Make some example time series data using a specified model

First make a series of images with simple numeric values from 1 to 30.

```javascript
// Define an arbitrary area of interest and create the imagery there
var aOI = ee.Geometry.Point([8.548333, 47.374722]).buffer(1000).bounds();
var projection = ee.Projection("EPSG:4326").atScale(100);
var numImColl = ee.ImageCollection(ee.List.sequence(1, 30)
                .map(function(n) {return ee.Image.constant(n)
                        .cast({"constant": ee.PixelType.int8()}, ["constant"])
                        .set('num', n).rename('num')
                        .addBands(ee.Image.constant(n).rename('time'))
                        .clip(aOI).reproject(projection)}))
                        .cast({'num': 'float','time': 'float'}, ['num', 'time']);
Map.centerObject(aOI,13);
```

Next, use this numeric series of images and choose an arbitrary model $a * log(n + b) + c$ (where $a=2$, $b=1$, and $c=3$) to prepare an example time series of data (generated from the mathematical model with random error applied).

```javascript
// Describe the model as an expression
var expOfChoice = "2 * log(b('num') + 1) + 3";

// Apply the algebraic expression to the image collection
var expressionApplied = numImColl.map(function(i) {
    return ee.Image(i.select('num').expression(expOfChoice).rename('num')
        .copyProperties(i)).addBands(i.select('time'));
});

// Apply some random error to the series so it's ready to model
// as an example dataset
var maxErrorDecimal = 0.1;
var computeError = function(i) {
    var seed = ee.Number(ee.Image(i).get('num'));
    var randomField = ee.Image.random(seed, 'normal');
    var maxError = i.select('num').multiply(ee.Image.constant(maxErrorDecimal));
    var finalError = maxError.multiply(randomField);
    var imageToCast = i.select('num').add(finalError).rename('numWithError').addBands(i.select('time'));
    return ee.Image(imageToCast).float();
};

var timeSeries = expressionApplied.map(computeError)
                                  .map(function(i){return i.set('system:footprint',aOI)});
Map.addLayer(timeSeries,{},'Example Time Series');
```

### Input the necessary parameters to optimize the model of interest

You'll need to ensure that all of your function inputs are properly formatted. You'll find a detailed description of the variable inputs below the demo.

```javascript
// Import geeode
var geeode = require('users/uzheoas/geeode:geeode.js');

// Input all necessary parameters

// For a quick demo, use only 10 population members and 10 iterations
var pop_size = 10;
var iNum = 10;

// Convert the arbitrary model above to expression format
var funToOpt = "b('a') * log(b('time') + b('b')) + b('c')";

// Input the name of the variables being optimized and the
// known numeric bounds of the variables
var inputVars = ['a','b','c'];
var inputBounds = [
    [0, 10],
    [0, 10],
    [0, 10]
];

// Input the name of the band being modelled (the dependent variable)
var bandName = "numWithError";
```

### Call the function

One you've confirmed all inputs are correctly formatted, call the function.

```javascript
// Call the function
var deOptimOutput = geeode.de_optim(pop_size,
                                    iNum,
                                    funToOpt,
                                    inputVars,
                                    inputBounds,
                                    timeSeries,
                                    bandName);
print('DE Optim Output', deOptimOutput);
Map.addLayer(deOptimOutput, {}, "DE Optim Output", false);
```

### Use the predicted parameters to visualize the model

Once you have a predicted value from the function you can apply it to the time series using the provided function.

```javascript
var collWithPredictions = geeode.applyModel(timeSeries, deOptimOutput, funToOpt)
Map.addLayer(collWithPredictions, {}, 'collWithPredictions', false);
```

### Examine the outputs

[Here's](https://code.earthengine.google.com/d8084b4a4ac9549e2a5a0227125b2955) what it looks like when you put it all together. Click around the image to inspect the pixel values of the predicted variables `a`,`b`, and `c` after 10 iterations on 10 population members.

The output "coefficients image" will contain the same number of bands (with the same names) as the bands/variables you're optimizing within your model. There will also be an `RMSE` band that contains the root mean square error of the model.

### Real World Example

You can find a real world example on an Sentinel-2 NDVI time series [here](https://code.earthengine.google.com/9771b48d1c6ec0358ed377ec49dfe771).
