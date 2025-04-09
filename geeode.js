exports.de_optim = function(pop_size,
                            iNum,
                            funToOpt,
                            inputVars,
                            inputBounds,
                            timeSeries,
                            bandName,
                            optParams) {
    
    // !! Before anything else, perform default value setting and type checking
    // !! to assist users with inputting the proper values
    
    // Set optional values to defaults if they are not explicitly defined
    if (optParams === undefined) {
      optParams = {};
    }
    
    var timeBand = optParams.timeBand;
    if (timeBand === undefined) {
      timeBand = 'time';
    }
    
    var mutationStrategy = optParams.mutationStrategy;
    if (mutationStrategy === undefined) {
      mutationStrategy = "rand";
    }
    
    var F = optParams.F;
    if (F === undefined) {
      F = 0.5;
    }
    
    var cr = optParams.cr;
    if (cr === undefined) {
      cr = 0.7;
    }
    
    var computeScree = optParams.computeScree;
    if (computeScree === undefined) {
      var computeScree = false;
    }
    
    var initialPopSeed = optParams.initialPopSeed;
    if (initialPopSeed === undefined) {
      var initialPopSeed = 1;
    }
    
    var parallelScale = optParams.parallelScale;
    if (parallelScale === undefined) {
      var parallelScale = 1;
    }
    
    var daisyChain = optParams.daisyChain;
    if (daisyChain === undefined) {
      var daisyChain = false;
    }
    
    var startingPopulationImage = optParams.startingPopulationImage;
    if (startingPopulationImage === undefined) {
      var startingPopulationImage = null;
    }
    
    var existingScreeImage = optParams.existingScreeImage;
    if (existingScreeImage === undefined) {
      var existingScreeImage = null;
    }
    
    // Check that the function is formatted correctly
    var propBandRegEx = "b\\('[a-zA-Z]+'\\)";
    var varPreRegEx = "b\\('";
    var varSuffRegEx = "'\\)";
    
    if (funToOpt.match(propBandRegEx) == null) {
      throw new Error("Your function must include all of the bands you would like to analyze in the appropriate format (including your specified time band); i.e., b('bandname').");
    }
    
    var allVarsList = inputVars.concat([timeBand]);
    var matchList = ee.String(funToOpt).match(propBandRegEx,'g').distinct()
                    .map(function(s){return ee.String(s).replace(varPreRegEx,'')
                                                        .replace(varSuffRegEx,'');
                    });
    var matchOrNot = ee.List(allVarsList).sort().equals(matchList.sort());
    if (matchOrNot.getInfo() === false) {
      throw new Error("Make sure your inputted variable list and the variables in your inputted function match!");
    }
    
    // Check that input bounds are given for every variable
    if (inputVars.length != inputBounds.length) {
      throw new Error("You must supply a numeric bounds for each of the input variables. Ensure your order is correct!");
    }
    
    // Compute / retrieve an area of interest if producing a population image for daisy chaining
    if (daisyChain === true) {
      var aOI = ee.Geometry(ee.Image(timeSeries.first()).get('system:footprint'));
      if (aOI.getInfo() === null){
        throw new Error("Your time series lacks a proper 'system:footprint'. Set one for each image in the collection then retry the function!");
      }
    }
    
    // !! Begin the algorithm once inputs are checked
    
    // Format the variable dictionary with relevant information after sorting the zipped inputs
    var zipped = inputVars.map(function(e, i) {
        return [e, inputBounds[i]];
    }).sort();
    var varsList = zipped.map(function(value, index) {
        return value[0];
    });
    var boundsList = zipped.map(function(value, index) {
        return value[1];
    });
    var varsDict = ee.Dictionary.fromLists(varsList, boundsList);

    // Create an initial population array image of candidate solutions;
    // Initialize an array image with the appropriate population number
    // wherein every population member vector is the appropriate size
    var numOfVarsToOpt = ee.Dictionary(varsDict).size();
    var boundsDictKeys = ee.List(varsList);
    var boundsArray = ee.Dictionary(varsDict).toArray(varsList);
    var boundsRangesArray = varsDict.map(function(k, v) {
        return ee.Number(ee.List(v).get(1)).subtract(ee.Number(ee.List(v).get(0)));
    }).toArray(varsList).repeat(1, pop_size).transpose(1, 0);
    var boundsMinsArray = varsDict.map(function(k, v) {
        return ee.Number(ee.List(v).get(0));
    }).toArray(varsList).repeat(1, pop_size).transpose(1, 0);
    var arrayImageToMultiply = ee.Image(boundsRangesArray).toArray();
    var arrayImageToAdd = ee.Image(boundsMinsArray).toArray();
    var numElements = boundsArray.reduce('count', [0]).reduce('sum', [1])
        .project([0]).get([0]);
    var reshapeImage = ee.Image([pop_size, numOfVarsToOpt]).toArray();
    var initialPopImage = ee.ImageCollection(ee.List.sequence(1, ee.Number(pop_size).multiply(boundsDictKeys.size()))
            .map(function(n) {
                return ee.Image.random(ee.Number(n).add(ee.Number(initialPopSeed).multiply(1e4)));
            }))
        .toArrayPerBand()
        .arrayReshape(reshapeImage, 2)
        .multiply(arrayImageToMultiply).add(arrayImageToAdd);

    // ~~~~~~~~~~~~~~~~
    // Iterate through the algorithm the instructed number of times
    // ~~~~~~~~~~~~~~~~

    // Make an iteration number collection that can be mapped as an 
    // image/feature collection; first, acquire the starting seed number
    if (startingPopulationImage === null) {
        var startingSeedNumber = 0;
    } else {
        var startingSeedNumber = ee.Number(ee.Image(startingPopulationImage).get('seed_num'));
    }
    var makeIterArray = function(n) {
        var a = [];
        var start = 1 + startingSeedNumber;
        var end = n + startingSeedNumber;
        for (var i = start; i <= end; i++) {
            a.push(i);
        }
        return a;
    };
    var iterJSArray = makeIterArray(iNum);
    var iterColl = ee.ImageCollection(
        iterJSArray.map(function(n) {
            return ee.Image.constant(n).set('seed_num', ee.Number(n));
        })
    );

    // Take count of how many scree bands there are (if the image is provided)
    if (existingScreeImage === null) {
        var numExtantScreeBandsPlus1 = 1;
    } else {
        var numExtantScreeBandsPlus1 = ee.Number(ee.Image(existingScreeImage).bandNames().length()).add(1);
    }

    // Make a population number collection that can be mapped as an 
    // image/feature collection
    var makeArray = function(n) {
        var a = [];
        for (var i = 0; i <= n - 1; i++) {
            a.push(i);
        }
        return a;
    };
    var popsizeJSArray = makeArray(pop_size);
    var popsizeColl = ee.ImageCollection(
        popsizeJSArray.map(function(n) {
            return ee.Image.constant(n).set('iter_num', ee.Number(n));
        })
    );

    // Make RMSE required functions
    // First, make a string for computing RMSE via the .expression() function;
    // this string will be used in multiple functions computing RMSE
    var beforeString = ee.String("(b('");
    var afterString = ee.String("') - b('modeled')) ** 2");
    var desiredString = beforeString.cat(bandName).cat(afterString);
    
    // Make a function that computes the summed/total RMSE for a collection
    var computeRMSEForColl = function(collToUse, imageToAssess) {
        // Map a function across the inputted collection to make a modeled layer,
        // then sum the residuals and compute RMSE
        var modeledValue = collToUse.map(function(i) {
            var bandsAdded = i.addBands(imageToAssess);
            var modeledValue = bandsAdded.addBands(bandsAdded.expression(funToOpt).rename('modeled'));
            return modeledValue.addBands(modeledValue.expression(desiredString).rename('RMSE'));
        });
        var meanRMSE = ee.Image(modeledValue.select('RMSE').reduce('mean', parallelScale)).rename('mean');
        var finalRMSE = meanRMSE.sqrt().rename('RMSE');
        return finalRMSE;
    };

    // Add RMSE to each set of potential coefficients using the time series
    var addRMSE = function(collWithCoeffs, timeSeries) {
        // Map a function across the inputted collection to make a modeled layer,
        // then sum the residuals and compute RMSE
        var rmseImages = collWithCoeffs.map(function(i) {
            var tsWithSquaredResids = timeSeries.map(function(tsi) {
                var tsWithBands = tsi.addBands(i);
                var tsWithPredictions = tsWithBands.addBands(tsWithBands.expression(funToOpt).rename('modeled'));
                var tsWithSquaredResids = i.addBands(tsWithPredictions.expression(desiredString).rename('RMSE'));
                return tsWithSquaredResids;
            });
            var meanRMSE = ee.Image(tsWithSquaredResids.select('RMSE').reduce('mean', parallelScale)).rename('mean');
            var finalRMSE = meanRMSE.sqrt().rename('RMSE');
            var inverseRMSE = ee.Image.constant(1).divide(finalRMSE).rename('inverseRMSE');
            return i.addBands(finalRMSE).addBands(inverseRMSE);
        });
        return rmseImages;
    };

    // Make a function that accepts an initial population image as its main
    // argument maps through the iteration collection and choose candidate
    // vectors randomly, then compare them with the trial vector and choose
    // the best candiate via a crossover function;
    
    // For this function, definte the available mutation strategies in a list
    // !! Adjust this last whenever augmenting the available strategies in the
    // !! mutatePopulation() function below.
    var strategyList = ["rand", "best"];
    
    if (strategyList.indexOf(mutationStrategy) == -1) {
    print('Invalid mutation strategy chosen. Choose one of:');
    print(strategyList);
    throw new Error('Input one of the suggested strategies (as a string object).') ;
    }
    
    var mutatePopulation = function(iPI, seedFactor, screeBandNum) {
        var mutationColl = popsizeColl.map(function(i) {
            // Acquire the iteration number
            var iterNum = ee.Number(ee.Image(i).get('iter_num'));
            var iterNumPlus1 = iterNum.add(1);

            // Use the number to remove the nth vector in the array
            var firstSlice = ee.Image(iPI).arraySlice(0, 0, iterNum, 1);
            var secondSlice = ee.Image(iPI).arraySlice(0, iterNumPlus1, pop_size, 1);
            var arrayImageMinusN = firstSlice.arrayCat(secondSlice, 0);

            // Also, save the nth vector from the array
            var mutationNVector = ee.Image(iPI).arraySlice(0, iterNum, iterNumPlus1);

            // Shuffle the array and slice off the top 3 members
            var reshapeKeysImage = ee.Image([pop_size - 1, 1]).toArray();
            var arrayAsIC = makeArray(pop_size - 1).map(function(s) {
                return ee.Image(0).set('aN', s);
            });
            var arrayKeys = ee.ImageCollection(arrayAsIC.map(function(s) {
                return ee.Image.random(ee.Number(ee.Image(s).get('aN')).add(seedFactor).add(iterNum.multiply(1e3)));
            })).toArrayPerBand().arrayReshape(reshapeKeysImage, 2);
            var shuffled = arrayImageMinusN.arraySort(arrayKeys);
            var sliced = shuffled.arraySlice(0, 0, 3);

            // Mutation operation to make a mutated vector

            // Mutation: rand
            var array0 = sliced.arraySlice(0, 0, 1);
            var array1 = sliced.arraySlice(0, 1, 2);
            var array2 = sliced.arraySlice(0, 2, 3);
            var fArray = ee.Image(F).toArray().arrayRepeat(1, numOfVarsToOpt);
            var mutationOpt1 = fArray.multiply(array1.subtract(array2));
            var mutatedVector_DE_Rand_1 = array0.add(mutationOpt1);

            // Mutation: best
            // Add RMSE to each set of potential coefficients using the global time series variable
            // Make a function to add RMSE to each set of potential coefficients using the time series
            var addRMSEToImage = function(i) {
                var tsWithSquaredResids = timeSeries.map(function(tsi) {
                    var tsWithBands = tsi.addBands(i);
                    var tsWithPredictions = tsWithBands.addBands(tsWithBands.expression(funToOpt).rename('modeled'));
                    var tsWithSquaredResids = i.addBands(tsWithPredictions.expression(desiredString).rename('RMSE'));
                    return tsWithSquaredResids;
                });
                var meanRMSE = ee.Image(tsWithSquaredResids.select('RMSE').reduce('mean', 2)).rename('mean');
                var finalRMSE = meanRMSE.sqrt().rename('RMSE');
                var inverseRMSE = ee.Image.constant(1).divide(finalRMSE).rename('inverseRMSE');
                return i.addBands(finalRMSE).addBands(inverseRMSE);
            };

            var iPIWithRMSE = popsizeColl.map(function(i) {
                // Acquire the iteration number
                var iterNum = ee.Number(ee.Image(i).get('iter_num'));
                var iterNumPlus1 = iterNum.add(1);
                // Save the nth array
                var mutationNVector = ee.Image(iPI).arraySlice(0, iterNum, iterNumPlus1);
                // Flatten to an image
                var imageToReturn = mutationNVector.arrayProject([1]).arrayFlatten([inputVars]);
                return imageToReturn;
            }).map(addRMSEToImage);
            var arrayToSortForBest = iPIWithRMSE.select(varsList).toArray();
            var keysForArraySortForBest = iPIWithRMSE.select('RMSE').toArray();
            var sortedArrayForBest = arrayToSortForBest.arraySort(keysForArraySortForBest);
            var bestVector = sortedArrayForBest.arraySlice(0, 0, 1);
            var mutatedVector_DE_Best_1 = bestVector.add(mutationOpt1);

            // Return the desired mutated vector
            var availableStrategies = ee.List(strategyList);
            var mutationStrategyInput = ee.String(mutationStrategy);
            var chosenStrategyValue = ee.Algorithms.If(mutationStrategyInput.equals("rand"), mutatedVector_DE_Rand_1,
                ee.Algorithms.If(mutationStrategyInput.equals("best"), mutatedVector_DE_Best_1, "NA"));
            var mutatedVector = chosenStrategyValue;

            // Convert to a multiband image for bounds clipping
            var mutatedVectorMultiband = ee.Image(mutatedVector).arrayProject([1]).arrayFlatten([varsList]);
            var minBoundsMultiband = varsDict.map(function(k, v) {
                return ee.Number(ee.List(v).get(0));
            }).toImage().select(varsList);
            var maxBoundsMultiband = varsDict.map(function(k, v) {
                return ee.Number(ee.List(v).get(1));
            }).toImage().select(varsList);
            var clippedMutatedVectorMultiband = mutatedVectorMultiband.where(mutatedVectorMultiband.lt(minBoundsMultiband), minBoundsMultiband)
                .where(mutatedVectorMultiband.gt(maxBoundsMultiband), maxBoundsMultiband);
            var clippedMutatedVectorArrayImage = clippedMutatedVectorMultiband.toArray().arrayReshape(ee.Image([1, numOfVarsToOpt]).toArray(), 2);

            // Create an image of crossover random values for the crossover binomial trials
            var varsLengthArrayAsIC = makeArray(varsList.length).map(function(s) {
                return ee.Image(0).set('varS', s);
            });
            var crossOverRandomValue = ee.ImageCollection(varsLengthArrayAsIC.map(function(c) {
                return ee.Image.random(ee.Number(ee.Image(c).get('varS')).add(seedFactor).add(iterNum.multiply(1e3)));
            })).toArrayPerBand().arrayFlatten([varsList]);

            var crImage = ee.Image(makeArray(varsList.length).map(function(n) {
                return ee.Image(cr).rename(varsList[n]);
            }));

            // Create a trial vector multiband image
            var targetVectorMultiband = mutationNVector.arrayProject([1]).arrayFlatten([varsList]);

            // Apply the crossover function
            var trialVectorMultiband = targetVectorMultiband.where(crossOverRandomValue.lt(crImage), clippedMutatedVectorMultiband);

            // Apply the objective function to the target and trial images
            var targetObjApplied = computeRMSEForColl(timeSeries, targetVectorMultiband);
            var trialObjApplied = computeRMSEForColl(timeSeries, trialVectorMultiband);

            var finalVectorToStore = targetVectorMultiband.where(trialObjApplied.lt(targetObjApplied), trialVectorMultiband);

            return finalVectorToStore;
        });
        var mutationArray = mutationColl.toBands().toArray().arrayReshape(ee.Image([pop_size, numOfVarsToOpt]).toArray(), 2);

        // Convert the array to a collection for RMSE calculation
        var icForRMSECalc = popsizeColl.map(function(i) {
            // Acquire the iteration number
            var iterNum = ee.Number(ee.Image(i).get('iter_num'));
            var iterNumPlus1 = iterNum.add(1);
            // Save the nth array
            var mutationNVector = ee.Image(mutationArray).arraySlice(0, iterNum, iterNumPlus1);
            // Flatten to an image
            var imageToReturn = mutationNVector.arrayProject([1]).arrayFlatten([varsList]);
            return imageToReturn;
        });

        var finalScreeColl = addRMSE(icForRMSECalc, timeSeries);

        // Calculate the best/lowest RMSE
        var finalRMSEImage = ee.Image(finalScreeColl.select('RMSE').reduce('min'));
        var screeBandsToAdd = ee.Image(ee.Image(iPI).get('scree'))
            .addBands(finalRMSEImage.rename(ee.String('RMSE_').cat(ee.Number.parse(screeBandNum).add(numExtantScreeBandsPlus1).format('%03d'))));
        return mutationArray.set('scree', ee.Image(screeBandsToAdd)).set('seed_num', ee.Number(seedFactor).divide(1e12));
    };

    // Instantiate a starting image for the iteration process
    var startingImageForIterate = initialPopImage.set('seed_num', 0).set('scree', ee.Image([]));

    // According to the two image arguments, accept pre-defined population and scree images
    if (startingPopulationImage === null && existingScreeImage === null) {
        var startingImageForIterate = initialPopImage.set('seed_num', 0).set('scree', ee.Image([]));
    } else if (startingPopulationImage === null && existingScreeImage != null) {
        var screeImageForIterate = ee.Image(existingScreeImage);
        var startingImageForIterate = initialPopImage.set('seed_num', 0).set('scree', screeImageForIterate);
    } else {
        var screeImageForIterate = ee.Image(existingScreeImage);
        var startingImageForIterate = ee.Image(startingPopulationImage).set('scree', screeImageForIterate);
    }

    // Iterate the function according to the specified number of times
    var populationOutput = iterColl.iterate(function(current, result) {
        return mutatePopulation(result, ee.Number(current.get('seed_num')).multiply(1e12), ee.Number(current.get('system:index')));
    }, startingImageForIterate);
    var screeImage = ee.Image(ee.Image(populationOutput).get('scree'));

    var flattenedIC = popsizeColl.map(function(i) {
        // Acquire the iteration number
        var iterNum = ee.Number(ee.Image(i).get('iter_num'));
        var iterNumPlus1 = iterNum.add(1);
        // Save the nth array
        var mutationNVector = ee.Image(populationOutput).arraySlice(0, iterNum, iterNumPlus1);
        // Flatten to an image
        var imageToReturn = mutationNVector.arrayProject([1]).arrayFlatten([varsList]);
        return imageToReturn;
    });

    var finalColl = addRMSE(flattenedIC, timeSeries);

    // Use quality mosaic to determine the best chromosome
    var mosaicedImage = finalColl.qualityMosaic('inverseRMSE');

    // Return the parameters or the scree plot depending on what is requested
    if (daisyChain === false) {
        print("DE Optim is returning a:");
        if (computeScree === true) {
            print("Scree Image");
            return ee.Image(screeImage);
        } else {
            print("Coefficients Image");
            return mosaicedImage.select(mosaicedImage.bandNames().remove('inverseRMSE'));
        }
    } else {
        print("DE Optim is returning a:");
        if (computeScree === false) {
            print("Population Image");
            return ee.Image(populationOutput).clip(aOI);
        } else {
            print("Scree Image");
            return ee.Image(screeImage);
        }
    }
};
