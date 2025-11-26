// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


// de_optim
/**
 * Perform differential‑evolution optimization on an Earth Engine time‑series.
 *
 * @param {number} pop_size - the number of population members when generating a new population
 * @param {number} iNum - the number of iterations
 * @param {string} funToOpt - the function you want to optimize in the form of a GEE expression; the function must include the time band as well as all variables you'll be optimizing; band names must be formatted using the prescribed GEE format `b('...')` 
 * @param {string[]} inputVars - a list of the names of the variables you will optimize
 * @param {Array<Array<number>>} inputBounds - a list of lists; each sublist describes the numeric bounds (inclusive) that will be used for each variable; the order must match the `inputVars` list
 * @param {ee.ImageCollection} timeSeries - the image collection time series being modelled; each image must have the raw data being modelled as a band name (identified below) as well as a time band and a properly formatted `system:footprint`
 * @param {string} bandName - the band name of the raw data being modelled; each image in the time series must contain this band
 * @param {Object} [optParams] - optional dictionary of additional parameters; see "Other Parameters" below
 *
 * @param {string} [optParams.timeBand='time'] - name of the time band:
 * @param {string} [optParams.mutationStrategy='rand'] - mutation strategy; possible values are `rand` and `best`
 * @param {number} [optParams.M=0.5] - mutation factor $M in (0,2]$ within the function $v_x + M * (v_y – v_z)$ where $v_x$,$v_y$, and $v_z$ are population members
 * @param {number} [optParams.cr=0.7] - crossover factor for the binomial selection process $cr in (0,1)$
 * @param {boolean} [optParams.computeScree=false] - `true` if you want to return a scree image, else `False`
 * @param {number} [optParams.initialPopSeed=1] - the seed used to generate the initial population
 * @param {number} [optParams.parallelScale=1] - the `parallelScale` value used to input into relevant EE functions; $in {1 ... 16}$
 * @param {boolean} [optParams.daisyChain=false] - `true` if you want to return the final iteration population image (i.e., an array-image), else `False`
 * @param {ee.Image} [optParams.startingPopulationImage=null] - the `ee.Image` object to input as a starting population image; must have a Integer value property `seedNum` and must be an array-image formatted according to the variables being optimized
 * @param {ee.Image} [optParams.existingScreeImage=null] - the `ee.Image` image to input as a starting scree image; should be an object computed from `computeScree`
 * @param {boolean} [optParams.verbosePrinting=false] - if `true`, print info upong running
 *
 * @returns {ee.Image|ee.ImageCollection} Placeholder describing the returned object (e.g., coefficients image, scree plot, or population image).
 *
 * @throws {Error} If required inputs are missing or malformed (e.g., mismatched variable lists, missing bounds, or invalid mutation strategy).
 *
 * @example
 * var result = deOptim({
 *   pop_size: 10,
 *   iNum: 25,
 *   funToOpt: "b('time') * b('a') + b('b')",
 *   inputVars: ["a", "b"],
 *   inputBounds: [[0, 1], [0, 10]],
 *   timeSeries: time_series,
 *   bandName: "NDVI"
 * });
 */
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

    var M = optParams.M;
    if (M === undefined) {
        M = 0.5;
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
    
    var verbosePrinting = optParams.verbosePrinting;
    if (verbosePrinting === undefined) {
        var verbosePrinting = false;
    }

    // Check that the function is formatted correctly
    var propBandRegEx = "b\\('[a-zA-Z]+'\\)";
    var varPreRegEx = "b\\('";
    var varSuffRegEx = "'\\)";

    if (funToOpt.match(propBandRegEx) == null) {
        throw new Error("Your function must include all of the bands you would like to analyze in the appropriate format (including your specified time band); i.e., b('bandname').");
    }

    var allVarsList = inputVars.concat([timeBand]);
    var matchList = ee.String(funToOpt).match(propBandRegEx, 'g').distinct()
        .map(function(s) {
            return ee.String(s).replace(varPreRegEx, '')
                .replace(varSuffRegEx, '');
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
        if (aOI.getInfo() === null) {
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
        throw new Error('Input one of the suggested strategies (as a string object).');
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
            var fArray = ee.Image(M).toArray().arrayRepeat(1, numOfVarsToOpt);
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
    } else if (startingPopulationImage != null && existingScreeImage === null) {
        var startingImageForIterate = ee.Image(startingPopulationImage).set('scree', ee.Image([]));
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
        if (verbosePrinting === true) {
            print("DE Optim is returning a:");
        }
        if (computeScree === true) {
            if (verbosePrinting === true) {
              print("Scree Image");
            }
            return ee.Image(screeImage);
        } else {
            if (verbosePrinting === true) {
              print("Coefficients Image");
            }
            return mosaicedImage.select(mosaicedImage.bandNames().remove('inverseRMSE'));
        }
    } else {
        if (verbosePrinting === true) {
            print("DE Optim is returning a:");
        }
        if (computeScree === false) {
            if (verbosePrinting === true) {
              print("Population Image");
            }
            return ee.Image(populationOutput);
        } else {
            if (verbosePrinting === true) {
              print("Scree Image");
            }
            return ee.Image(screeImage);
        }
    }
};


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// sub_sample
/**
 * Subsamples an image collection by temporal density.
 *
 * @param {ee.ImageCollection} iC - the GEE image collection to subsample
 * @param {number} nKeep - the number of observations to keep at every pixel (maximum)
 * @param {string} sType - the subsampling type you'd like to perform; one of `'bulk'`, `'splitshuffle'`, `'leapfrog'`
 * @param {string} bandName - the name of the band (in the image collection) containing your value of interest
 *
 * @param {number} [nStD=0.5] - the number of standard deviations to use as a kernel width when calculating temporal density
 * @param {string} [timeBandName='time'] - the name of the time band in each image
 * @param {number} [sN=4] - the number of splits if using the `'splitshuffle'` method
 * @param {number} [seedNum=1] - the random seed used for shuffling
 * @param {boolean} [verbosePrinting=false] - if `true`, print info upon running
 *
 * @returns {ee.Image} An image comprised of `nKeep` paired bands; each pair of bands includes the time value and the original observed band values at that time
 *
 * @throws {Error} If required inputs are missing or malformed (e.g., `nStD` must be positive).
 *
 * @note Time data from the individual images in the input collection will be lost after using this function.
 *
 * @example
 * var sampledCollection = subSample({
 *   iC: yourImageCollection,
 *   nKeep: 30,
 *   sType: 'leapfrog',
 *   bandName: 'NDVI'
 * });
 */

exports.sub_sample = function(iC,
                         nKeep,
                         sType,
                         bandName,
                         optParams) {

    // Set optional values to defaults if they are not explicitly defined
    if (optParams === undefined) {
        optParams = {};
    }
    
    var nStD = optParams.nStD;
    if (nStD === undefined) {
        nStD = 0.5;
    } else if (nStD <= 0) {
    throw new Error('nStD must be greater than 0!');
    }
    
    var timeBandName = optParams.timeBandName;
    if (timeBandName === undefined) {
        timeBandName = 'time';
    }
    
    var sN = optParams.sN;
    if (sN === undefined) {
        sN = 4;
    } else if (sN <= 0) {
    throw new Error('sN must be greater than 0!');
    }
    
    var seedNum = optParams.seedNum;
    if (seedNum === undefined) {
        seedNum = 1;
    }
    
    var verbosePrinting = optParams.verbosePrinting;
    if (verbosePrinting === undefined) {
        verbosePrinting = false;
    }
    
    // Calculate the desired time standard deviation value
    var iCTimeStdDevHalf = iC.select('time').reduce(ee.Reducer.stdDev()).multiply(0.5);

    // Add ± min/max times (based on 0.5 StdDev) for temporal filtering
    var iCWithMinMax = iC.map(function(i) {
        return i.addBands(i.select('time').add(iCTimeStdDevHalf).rename('maxTime'))
            .addBands(i.select('time').subtract(iCTimeStdDevHalf).rename('minTime'));
    });

    // Filter based on ±min/max
    var densityColl = iCWithMinMax.map(function(o) {

        var densityBand = iCWithMinMax.map(function(i) {

            var testBand = i.select('time').gt(o.select('minTime'))
                .and(i.select('time').lt(o.select('maxTime')))
                .rename('density').updateMask(i.select(bandName).mask());

            var valueToSum = i.addBands(testBand);

            return valueToSum;

        }).select('density').sum();

        return o.addBands(densityBand.updateMask(o.select(bandName).mask()))
            .addBands(ee.Image.random(seedNum).add(1).rename('random').updateMask(o.select(bandName).mask()))
            .addBands(densityBand.multiply(ee.Image.random(seedNum).add(1)).rename('weight').updateMask(o.select('NDVI').mask()));
    });

    var keysShuffle = densityColl.select('weight').toArray();

    // Shuffle the time series values according to temporal density weight
    var originalTS = densityColl.select(bandName, 'time').toArray();
    var densityCollShuffled = originalTS.arraySort(keysShuffle);

    // Format array images that will serve as array masks, allowing for subsampled values to 
    // be returned or for their complement to be returned (i.e., the masked out values)
    var tsLength = densityCollShuffled.arrayLength(0);
    var nToRemoveDiff = ee.Image(tsLength).subtract(nKeep);
    var nToRemove = nToRemoveDiff.where(nToRemoveDiff.lte(ee.Image.constant(0)), 0);
    var onOffArray = ee.Image([1, 0]).toArray();
    var offOffArray = ee.Image([0, 0]).toArray();
    var onOffRepeated = onOffArray.arrayRepeat(0, nToRemove);
    var onOffRepeatedSliced = onOffRepeated.arraySlice(0, 0, tsLength);
    var onOffRepeatedSlicedLength = onOffRepeatedSliced.arrayLength(0);
    var onOffRepeatedKeepSlicedKeepSum = onOffRepeatedSliced.arrayReduce('sum', [0]);
    var additionalNumberToRemove = onOffRepeatedKeepSlicedKeepSum.multiply(ee.Image([0]))
        .where(onOffRepeatedKeepSlicedKeepSum.arrayFlatten([
                ['n']
            ]).subtract(ee.Image(nKeep)).gt(0),
            onOffRepeatedKeepSlicedKeepSum.subtract(nKeep));
    var onOffRepeatedLength = onOffRepeated.arrayLength(0);
    var offOffRepeated = offOffArray.arrayRepeat(0, additionalNumberToRemove.arrayFlatten([
        ['n']
    ]));
    var offOffRepeatedLength = offOffRepeated.arrayLength(0);

    // Format arrays that will be used if the default leap-frog sampling "laps" the
    // total number of samples (i.e., the sampling removes more than half of the original time series)
    var aggroSliceLength = onOffRepeatedSlicedLength.subtract(offOffRepeatedLength);
    var onOffRepeatedSlicedWithAggroSampling = onOffRepeatedSliced.arraySlice(0, 0, aggroSliceLength);
    var aggroMaskArray = onOffRepeatedSlicedWithAggroSampling.arrayCat(offOffRepeated, 0);
    var keepVector = ee.Image([1]).arrayRepeat(0, tsLength.subtract(onOffRepeatedLength));
    var normalMaskArray = keepVector.arrayCat(onOffRepeated, 0);

    // Finalize the mask to use for the time series
    var maskTest = onOffRepeatedLength.gt(tsLength);
    var maskArray = normalMaskArray.where(maskTest, aggroMaskArray);

    // Format the band names for the flattened image
    var nL = ee.List.sequence(1, nKeep).map(function(n) {
        return ee.String('b').cat(ee.Number(n).format('%02d'));
    });

    // Bulk sampling
    var bulkSamplingSliced = densityCollShuffled.arraySlice(0, 0, nKeep);
    var bulkSamplingTimeKeys = bulkSamplingSliced.arraySlice(1, -1);
    var bulkSampledArrayImage = bulkSamplingSliced.arraySort(bulkSamplingTimeKeys);
    var subSampledImageBulk = bulkSampledArrayImage.arrayPad([nKeep, 2]).arrayFlatten([nL, [bandName, 'time']], '_').selfMask();

    // Split shuffle sampling
    // Use the sN (number of splits) input to split the array pseudo randomly
    // (i.e., weighted by temporal density) and subsample them
    var nColl = ee.ImageCollection(ee.List.sequence(0, (sN - 1)).map(function(i) {
        return ee.Image.constant(i).set('n', i);
    }));
    var splitShuffledArrays = nColl.map(function(i) {
        return ee.Image(densityCollShuffled).arraySlice({
            axis: 0,
            start: i.int(),
            step: sN
        });
    }).toArrayPerBand();
    var densityCollSlicedSplitShuffled = splitShuffledArrays.arraySlice({
        axis: 0,
        end: ee.Image.constant(nKeep)
    });
    var keysTimeSplitShuffled = densityCollSlicedSplitShuffled.arraySlice(1, -1);
    var densityCollSortedSplitShuffled = densityCollSlicedSplitShuffled.arraySort(keysTimeSplitShuffled);
    var densityCollPaddedSplitShuffled = densityCollSortedSplitShuffled.arrayPad([nKeep, 2]);
    var subSampledImageSplitShuffled = densityCollPaddedSplitShuffled.arrayFlatten([nL, [bandName, 'time']], '_').selfMask();

    // Apply the array-mask to confirm the subsampling
    var densityCollMaskedLF = densityCollShuffled.arrayMask(maskArray.arrayRepeat(1, ee.Image(1)));
    var timeSortingKeysLF = densityCollMaskedLF.arraySlice({
        axis: 1,
        start: ee.Image.constant(-1)
    });
    var densityCollMaskedSortedLF = densityCollMaskedLF.arraySort(timeSortingKeysLF);

    // Mask the flattened array-image with itself to remove 0 values
    var subSampledImageLF = densityCollMaskedSortedLF.arrayPad([nKeep, 2]).arrayFlatten([nL, [bandName, 'time']], '_').selfMask();

    // Return the image collection of interest
    if (sType == 'bulk') {
        if (verbosePrinting === true) {
            print('Bulk Sampling');
        }
        var imageToReturn = subSampledImageBulk;
    } else if (sType == 'splitshuffle') {
        if (verbosePrinting === true) {
            print('Split Shuffle Sampling');
        }
        var imageToReturn = subSampledImageSplitShuffled;
    } else if (sType == 'leapfrog') {
        if (verbosePrinting === true) {
            print('Leapfrog Sampling');
        }
        var imageToReturn = subSampledImageLF;
    } else {
        throw new Error("Input one of: 'bulk', 'splitshuffle', or 'leapfrog'.");
        var imageToReturn = null;
    }

    return imageToReturn;
};


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ts_array_to_coll
/**
 * Reconstructs an image collection from a subsampled image where time information is stored as a band.
 *
 * @param {ee.Image} ts_image - the image outputted from the `sub_sample` function
 * @param {string} band_name - the original band name of the values being sampled
 * @param {number} ts_length - the number of samples maintained from the original time series per pixel
 *
 * @returns {ee.ImageCollection} an image collection where each image contains a `time` band and a band with the original sampled observation, ordered chronologically (earliest observation first)
 *
 * @example
 * var sampledCollection = subSample({
 *   i_c: yourImageCollection,
 *   n_keep: nObservationsToKeep,
 *   s_type: 'leapfrog',
 *   band_name: 'NDVI'
 * });
 *
 * var reconstructed = tsArrayToColl(sampledCollection, 'NDVI', nObservationsToKeep);
 */

exports.ts_array_to_coll = function(ts_image,
                                band_name,
                                ts_length) {
    
    // Make a function to create a client-side list of numbers
    var makeIterArray = function(s, n) {
        var a = [];
        var start = s;
        var end = n;
        for (var i = start; i <= end; i++) {
            a.push(i);
        }
        return a;
    };
    var iL = makeIterArray(0, ts_length - 1);

    // Make a function to pad the number with leading zeros
    function zeroPad(num, places) {
        var zero = places - num.toString().length + 1;
        return Array(+(zero > 0 && zero)).join("0") + num;
    }
    
    // Cracks open the image bands and return them in a form that the 
    // ee.ImageCollection.fromImages() function can handle
    var iC = iL.map(function(i) {
        var paddedNum = zeroPad(i + 1, 3);
        var main_band = "b" + zeroPad(i + 1, 2) + "_" + band_name;
        var timeBand = "b" + zeroPad(i + 1, 2) + "_time";

        return ee.Image(ts_image).select([main_band, timeBand], [band_name, 'time']);
    });

    // Create then return the ImageCollection
    return ee.ImageCollection.fromImages(iC);

};


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// apply_model
/**
 * Apply a model expression to each image in a collection.
 *
 * @param {ee.ImageCollection} timeSeries   the input time series ImageCollection on which to apply the model
 * @param {ee.Image} deOptimOutput          the coefficient image returned by `de_optim`
 * @param {string} funToOpt                 a string expression to evaluate; e.g., "b('time') * b('a') + b('b')"
 * @returns {ee.ImageCollection}            the collection with a new band named 'predicted' included
 */

exports.apply_model = function(timeSeries, deOptimOutput, funToOpt) {
  // Make a function that adds the coefficient bands, evaluates the expression,
  // renames the result, and adds the band back into the original image
  var applyExpression = function(image) {
    var predicted = image.addBands(deOptimOutput)
                         .expression(funToOpt)
                         .rename('predicted');

    // Return the original image with the new 'predicted' band included
    return image.addBands(predicted);
  };

  // Map the function over the whole collection
  var collWithPredictions = timeSeries.map(applyExpression);
  return collWithPredictions;
};