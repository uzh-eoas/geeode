## "Daisy-Chain"

For more in-depth analyses requiring a higher number of iterations on a population, make use of the "daisy-chain" functionality. "Daisy-chaining" refers to the process of using the outputted population from a previous run.

If you follow the [quickstart](./quickstart.md), continue with the following code block:

```javascript
// Call the function a first time and deliver to an export.
var firstDeOptimOutput = geeode.de_optim(pop_size,
                                         iNum,
                                         funToOpt,
                                         inputVars,
                                         inputBounds,
                                         timeSeries,
                                         bandName,
                                         {daisyChain:true});

Export.image.toAsset({
    image: firstDeOptimOutput,
    region: aOI,
    crs: 'EPSG:4326',
    scale: 10,
    description: 'daisy_chain_example',
});

// !! This image asset ID should be updated to your own project space.
var daisyChainPopulation = ee.Image('projects/uzheoas/assets/daisy_chain_example');

// Call the function a second time with the daisy-chained population
var finalDeOptimOutput = geeode.de_optim(pop_size,
                                         iNum,
                                         funToOpt,
                                         inputVars,
                                         inputBounds,
                                         timeSeries,
                                         bandName,
                                         {startingPopulationImage:daisyChainPopulation});
print('Final DE Optim Output', finalDeOptimOutput);
Map.addLayer(finalDeOptimOutput, {}, "DE Optim Output", false);

var collWithPredictions = geeode.apply_model(timeSeries, finalDeOptimOutput, funToOpt);
Map.addLayer(collWithPredictions, {}, 'collWithPredictions', false);
```

You can see the example in the GEE Code Playground [here](https://code.earthengine.google.com/4ab865e3add8ec6a6c2ad7bad948044e).

There are 2 specific attributes that an inputted population image must fulfill:

- The image must have a property named `seed_num` with an integer value specified;
- The dimensions of the embedded array-image must meet dimensionality contraints; in particular:
    - the number of rows in each pixel-array must equal the specified `pop_size`
    - the number of columns in each pixel-array must equal the number of inputted variables being optimized

## Custom Population Construction

Due to the design of the algorithm, users can begin their workflow with a 
randomly generated set of functions within an initial population and proceed 
the evolution process. Or, instead of creating a new population of potential
models, users can begin by creating a custom set of population members.

These custom populations can be specified in whatever fashion users choose so 
long as they match the required format for the function being optimized (i.e., 
their array dimensionality at the pixel level must match the number of 
coefficients being optimized and the specified `pop_size`).

### "Divide-and-Conquer"

One approach to custom populations is to begin a workflow by running multiple 
sets of randomly generated populations, each being generated with different 
random seeds. The results of these populations can then be stacked together 
themselves to create a population that can be evolved. Notably, stacks of 
coefficient images from any number of `de_optim` runs can easily be combined 
into new populations:

```python

# Take a collection of combined de_otpim outputs
chromoColl = ee.ImageCollection([first_deoptim_output_image,second_deoptim_output_image,third_deoptim_output_image])
chromoBandNames = ee.Image(chrompColl.first()).bandNames();

# This new array-image can now be used as an initial population for a new de_optim run.
chromoArray = chromColl.select(chromoBandNames.removeAll(['RMSE'])).toArray().set('seed_num',newSeed);

```

## Helper Functions

### "Pause-and-Wait" with Asset/Task Checks

In order to assist with constructing more in-depth workflows, especially in 
situations where daisy-chaining is necessary, `geeode` offers functions both 
to easily check whether a given asset already exists (or is running as an asset)
pause the workflow operation until a particular task has completed.

```python
result = de_optim(
    pop_size=10,
    iNum=25,
    funToOpt="b('time') * a + b",
    inputVars=["a", "b"],
    inputBounds=[[0, 1], [0, 10]],
    timeSeries=time_series,
    bandName="NDVI",
)

task = ee.batch.Export.image.toAsset(
    image = result,
    assetId = aID,
    description = 'Task_Description'
    ...<other_parameters>
);
task.start()

# Check for the asset then issue the task if it doesn't already exist
check_for_asset_then_run_task(aID,task,'Task_Description')

# Pause the workflow and wait until the task has finished before continuing
pause_and_wait('Task_Description')

```

These functions give users the ability to write any number of steps to their 
workflows and run them from start to finish without required interactivity.


### Adding Time as a Band

If you don't currently have a time variable in your image collection as a band, 
consider a function like this one (to add it as a fractional year value at every
pixel).

```javascript
// Add Fractional Time to each image pixel
var addFractionalTime = function(i) {
    var yearFraction = ee.Image(i).date().getFraction('year');
    var yearInteger = ee.Image(i).date().get('year');
    var fractionalYear = ee.Image.constant(yearInteger.add(yearFraction)).rename('time');
    return i.addBands(fractionalYear);
}

```