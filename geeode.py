#!/usr/bin/env python
import re
import ee

def de_optim(pop_size = int,
             iNum = int ,
             funToOpt = str,
             inputVars = list[str],
             inputBounds = list[list[float]],
             timeSeries = ee.ImageCollection,
             bandName = str,
             optParams = {}):
    
    # !! Before anything else, perform default value setting and type checking
    # !! to assist users with inputting the proper values
    
    try:
        timeBand = optParams['timeBand']
    except KeyError:
        optParams['timeBand'] = 'time'
        timeBand = optParams['timeBand']
    
    try:
        mutationStrategy = optParams['mutationStrategy']
    except KeyError:
        optParams['mutationStrategy'] = 'rand'
        mutationStrategy = optParams['mutationStrategy']
    
    try:
        M = optParams['M']
    except KeyError:
        optParams['M'] = 0.5
        M = optParams['M']
    
    try:
        cr = optParams['cr']
    except KeyError:
        optParams['cr'] = 0.5
        cr = optParams['cr']
    
    try:
        computeScree = optParams['computeScree']
    except KeyError:
        optParams['computeScree'] = False
        computeScree = optParams['computeScree']
    
    try:
        initialPopSeed = optParams['initialPopSeed']
    except KeyError:
        optParams['initialPopSeed'] = 1
        initialPopSeed = optParams['initialPopSeed']
    
    try:
        parallelScale = optParams['parallelScale']
    except KeyError:
        optParams['parallelScale'] = 1
        parallelScale = optParams['parallelScale']
    
    try:
        daisyChain = optParams['daisyChain']
    except KeyError:
        optParams['daisyChain'] = False
        daisyChain = optParams['daisyChain']
    
    try:
        startingPopulationImage = optParams['startingPopulationImage']
    except KeyError:
        optParams['startingPopulationImage'] = None
        startingPopulationImage = optParams['startingPopulationImage']
    
    try:
        existingScreeImage = optParams['existingScreeImage']
    except KeyError:
        optParams['existingScreeImage'] = None
        existingScreeImage = optParams['existingScreeImage']

    # Check that the function is formatted correctly
    propBandRegEx = "b\\('[a-zA-Z]+'\\)";
    varPreRegEx = "b\\('";
    varSuffRegEx = "'\\)";
    matches = re.findall(propBandRegEx,funToOpt)

    if len(matches) == 0:
        raise ValueError("Your function must include all of the bands you would like to analyze in the appropriate format (including your specified time band); i.e., b('bandname').")

    allVarsList = inputVars + [timeBand];
    matches = re.findall(propBandRegEx,funToOpt)
    if len(matches) == 0:
        raise ValueError("Your function must include all of the bands you would like to analyze in the appropriate format (including your specified time band); i.e., b('bandname').")
    matches = [re.sub(varPreRegEx, '', s) for s in matches]
    matches = [re.sub(varSuffRegEx, '', s) for s in matches]
    if set(allVarsList) != set(matches):
        raise ValueError("Make sure your inputted variable list and the variables in your inputted function match!")

    # Check that input bounds are given for every variable
    if len(inputVars) != len(inputBounds):
      raise ValueError("You must supply a numeric bounds for each of the input variables. Ensure your order is correct!");

    # Compute / retrieve an area of interest if producing a population image for daisy chaining
    if daisyChain == True:
      aOI = ee.Geometry(ee.Image(timeSeries.first()).get('system:footprint'));
      if aOI.getInfo() == None:
        raise ValueError("Your time series lacks a proper 'system:footprint'. Set one for each image in the collection then retry the function!");

    # !! Begin the algorithm once inputs are checked
  
    # Format the variable dictionary with relevant information after sorting the zipped inputs
    sortedVarsBoundsDict = dict(sorted(dict(zip(inputVars,inputBounds)).items()))
    varsList = list(sortedVarsBoundsDict.keys())
    boundsList = list(sortedVarsBoundsDict.values())
    varsDict = ee.Dictionary.fromLists(varsList,boundsList);

    # Create an initial population array image of candidate solutions;
    # Initialize an array image with the appropriate population number
    # wherein every population member vector is the appropriate size
    numOfVarsToOpt = ee.Dictionary(varsDict).size();
    boundsDictKeys = ee.List(varsList);
    boundsArray = ee.Dictionary(varsDict).toArray(varsList);
    def forBoundsRangesArray(k,v):
        return ee.Number(ee.List(v).get(1)).subtract(ee.Number(ee.List(v).get(0)));
    boundsRangesArray = varsDict.map(forBoundsRangesArray).toArray(varsList).repeat(1,pop_size).transpose(1,0);
    def forBoundsMinsArray(k,v):
        return ee.Number(ee.List(v).get(0));
    boundsMinsArray = varsDict.map(forBoundsMinsArray).toArray(varsList).repeat(1,pop_size).transpose(1,0);
    arrayImageToMultiply = ee.Image(boundsRangesArray).toArray();
    arrayImageToAdd = ee.Image(boundsMinsArray).toArray();
    numElements = (boundsArray.reduce('count',[0]).reduce('sum',[1])
        .project([0]).get([0]));
    reshapeImage = ee.Image([pop_size,numOfVarsToOpt]).toArray();
    initialPopImage = (ee.ImageCollection(ee.List.sequence(1,ee.Number(pop_size).multiply(boundsDictKeys.size()))
            .map(lambda n: ee.Image.random(ee.Number(n).add(ee.Number(initialPopSeed).multiply(1e4)))))
        .toArrayPerBand()
        .arrayReshape(reshapeImage,2)
        .multiply(arrayImageToMultiply).add(arrayImageToAdd));

    # ~~~~~~~~~~~~~~~~
    # Iterate through the algorithm the instructed number of times
    # ~~~~~~~~~~~~~~~~

    # Make an iteration number collection that can be mapped as an 
    # image/feature collection; first, acquire the starting seed number
    if (startingPopulationImage == None):
        startingSeedNumber = 0;
    else:
        startingSeedNumber = int(ee.Number(ee.Image(startingPopulationImage).get('seed_num')).getInfo());
    def makeIterArray(n):
        return list(range(1+startingSeedNumber,n+1+startingSeedNumber))
    iterJSArray = ee.List(makeIterArray(iNum));
    iterColl = ee.ImageCollection(iterJSArray.map(lambda n: ee.Image.constant(n).set('seed_num',ee.Number(n))));
    
    # Take count of how many scree bands there are (if the image is provided)
    if existingScreeImage == None:
        numExtantScreeBandsPlus1 = 1;
    else:
        numExtantScreeBandsPlus1 = ee.Number(ee.Image(existingScreeImage).bandNames().length()).add(1);
    
    # Make a population number collection that can be mapped as an 
    # image/feature collection
    def makeArray(n):
        return list(range(0,n))
    popsizeJSArray = ee.List(makeArray(pop_size));
    popsizeColl = ee.ImageCollection(popsizeJSArray.map(lambda n: ee.Image.constant(n).set('iter_num',ee.Number(n))))

    # Make RMSE required functions
    # First, make a string for computing RMSE via the .expression() function;
    # this string will be used in multiple functions computing RMSE
    beforeString = ee.String("(b('");
    afterString = ee.String("') - b('modeled')) ** 2");
    desiredString = beforeString.cat(bandName).cat(afterString);

    # Make a function that computes the summed/total RMSE for a collection
    def computeRMSEForColl(collToUse,imageToAssess):
        # Map a function across the inputted collection to make a modeled NDVI layer,
        # then sum the residuals and compute RMSE
        def makeSquaredResiduals(i):
            bandsAdded = i.addBands(imageToAssess);
            modeledValue = bandsAdded.addBands(bandsAdded.expression(funToOpt).rename('modeled'));
            return modeledValue.addBands(modeledValue.expression("(b('NDVI') - b('modeled')) ** 2").rename('squaredResiduals'));
        modeledValue = collToUse.map(makeSquaredResiduals);
        meanRMSE = ee.Image(modeledValue.select('squaredResiduals').reduce('mean',parallelScale)).rename('mean');
        finalRMSE = meanRMSE.sqrt().rename('RMSE');
        return finalRMSE;
    
    # Add RMSE to each set of potential coefficients using the time series
    def addRMSE(collWithCoeffs,timeSeries):
        # Map a function across the inputted collection to make a modeled NDVI layer,
        # then sum the residuals and compute RMSE
        def makeRMSEImages(i):
            def makeSquaredResiduals(tsi):
                tsWithBands = tsi.addBands(i);
                tsWithPredictions = tsWithBands.addBands(tsWithBands.expression(funToOpt).rename('modeled'));
                tsWithSquaredResids = i.addBands(tsWithPredictions.expression("(b('NDVI') - b('modeled')) ** 2").rename('squaredResiduals'));
                return tsWithSquaredResids;
            tsWithRMSE = timeSeries.map(makeSquaredResiduals)
            meanRMSE = ee.Image(tsWithRMSE.select('squaredResiduals').reduce('mean',parallelScale)).rename('mean');
            finalRMSE = meanRMSE.sqrt().rename('RMSE');
            inverseRMSE = ee.Image.constant(1).divide(finalRMSE).rename('inverseRMSE');
            return i.addBands(finalRMSE).addBands(inverseRMSE);
        rmseImages = collWithCoeffs.map(makeRMSEImages);
        return rmseImages;
    
    # Make a function that accepts an initial population image as its main
    # argument maps through the iteration collection and choose candidate
    # vectors randomly, then compare them with the trial vector and choose
    # the best candiate via a crossover function;
    # !! Adjust this last whenever augmenting the available strategies in the
    # !! mutatePopulation() function below.
    strategyList = ["rand", "best"];

    try:
        strategyList.index(mutationStrategy)
    except ValueError:
        print('Invalid mutation strategy chosen. Choose one of:');
        print(strategyList);
        raise ValueError('Input one of the suggested strategies (as a string object).') ;
    
    def mutatePopulation(iPI,seedFactor,screeBandNum):
        def mutateCollection(i):
            # Acquire the iteration number
            iterNum = ee.Number(ee.Image(i).get('iter_num'));
            iterNumPlus1 = iterNum.add(1);
            
            # Use the number to remove the nth vector in the array
            firstSlice = ee.Image(iPI).arraySlice(0,0,iterNum,1);
            secondSlice = ee.Image(iPI).arraySlice(0,iterNumPlus1,pop_size,1);
            arrayImageMinusN = firstSlice.arrayCat(secondSlice,0);
            
            # Also, save the nth vector from the array
            mutationNVector = ee.Image(iPI).arraySlice(0,iterNum,iterNumPlus1);
            
            # Shuffle the array and slice off the top 3 members
            reshapeKeysImage = ee.Image([pop_size - 1,1]).toArray();
            arrayKeys = (ee.ImageCollection(ee.List(makeArray(pop_size - 1)).map(lambda s:
                ee.Image.random(ee.Number(s).add(seedFactor).add(iterNum.multiply(1e3)))
                ))
            .toArrayPerBand().arrayReshape(reshapeKeysImage,2));
            shuffled = arrayImageMinusN.arraySort(arrayKeys);
            sliced = shuffled.arraySlice(0,0,3);
            
            # Mutation operation to make a mutated vector

            # Mutation: rand
            array0 = sliced.arraySlice(0,0,1);
            array1 = sliced.arraySlice(0,1,2);
            array2 = sliced.arraySlice(0,2,3);
            fArray = ee.Image(M).toArray().arrayRepeat(1,numOfVarsToOpt);
            mutationOpt1 = fArray.multiply(array1.subtract(array2));
            mutatedVector_DE_Rand_1 = array0.add(mutationOpt1);

            # Mutation: best
            # Add RMSE to each set of potential coefficients using the global time series variable
            # Make a function to add RMSE to each set of potential coefficients using the time series
            def addRMSEToImage(i):
                def tsiFunction(tsi):
                    tsWithBands = tsi.addBands(i);
                    tsWithPredictions = tsWithBands.addBands(tsWithBands.expression(funToOpt).rename('modeled'));
                    tsWithSquaredResids = i.addBands(tsWithPredictions.expression(desiredString).rename('RMSE'));
                    return tsWithSquaredResids;
                tsWithSquaredResids = timeSeries.map(tsiFunction)
                meanRMSE = ee.Image(tsWithSquaredResids.select('RMSE').reduce('mean', 2)).rename('mean');
                finalRMSE = meanRMSE.sqrt().rename('RMSE');
                inverseRMSE = ee.Image.constant(1).divide(finalRMSE).rename('inverseRMSE');
                return i.addBands(finalRMSE).addBands(inverseRMSE);

            def iPIToRMSEHelper(i):
                # Acquire the iteration number
                iterNum = ee.Number(ee.Image(i).get('iter_num'));
                iterNumPlus1 = iterNum.add(1);
                # Save the nth array
                mutationNVector = ee.Image(iPI).arraySlice(0, iterNum, iterNumPlus1);
                # Flatten to an image
                imageToReturn = mutationNVector.arrayProject([1]).arrayFlatten([inputVars]);
                return imageToReturn;

            iPIWithRMSE = popsizeColl.map(iPIToRMSEHelper).map(addRMSEToImage)
            arrayToSortForBest = iPIWithRMSE.select(varsList).toArray();
            keysForArraySortForBest = iPIWithRMSE.select('RMSE').toArray();
            sortedArrayForBest = arrayToSortForBest.arraySort(keysForArraySortForBest);
            bestVector = sortedArrayForBest.arraySlice(0, 0, 1);
            mutatedVector_DE_Best_1 = bestVector.add(mutationOpt1);

            # Return the desired mutated vector
            availableStrategies = ee.List(strategyList);
            mutationStrategyInput = ee.String(mutationStrategy);
            chosenStrategyValue = ee.Algorithms.If(mutationStrategyInput.equals("rand"), mutatedVector_DE_Rand_1,
                ee.Algorithms.If(mutationStrategyInput.equals("best"), mutatedVector_DE_Best_1, "NA"));
            mutatedVector = chosenStrategyValue;
            
            # Convert to a multiband image for bounds clipping
            mutatedVectorMultiband = ee.Image(mutatedVector).arrayProject([1]).arrayFlatten([varsList]);
            def minBoundsToMultibandfunction(k,v):
                return ee.Number(ee.List(v).get(0));
            minBoundsMultiband = varsDict.map(minBoundsToMultibandfunction).toImage().select(varsList);
            def maxBoundsToMultibandfunction(k,v):
                return ee.Number(ee.List(v).get(1));
            maxBoundsMultiband = varsDict.map(maxBoundsToMultibandfunction).toImage().select(varsList);
            clippedMutatedVectorMultiband = (mutatedVectorMultiband
                                             .where(mutatedVectorMultiband.lt(minBoundsMultiband),minBoundsMultiband)
                                             .where(mutatedVectorMultiband.gt(maxBoundsMultiband),maxBoundsMultiband));
            
            # Create an image of crossover random values for the crossover binomial trials
            def makeCrossOver(c):
                return (ee.Image.random(ee.Number(c).add(seedFactor).add(iterNum.multiply(1e3))));
            crossOverRandomValue = (ee.ImageCollection(ee.List(makeArray(len(varsList)))
                                                                  .map(makeCrossOver))
                                    .toArrayPerBand().arrayFlatten([varsList]));
            
            crResult = []
            for v in varsList:
                crResult.append(cr)
            crImage = ee.Image(crResult).rename(varsList)
            
            # Create a trial vector multiband image
            targetVectorMultiband = mutationNVector.arrayProject([1]).arrayFlatten([varsList]);
            
            # Apply the crossover function
            trialVectorMultiband = targetVectorMultiband.where(crossOverRandomValue.lt(crImage),clippedMutatedVectorMultiband);
            
            # Apply the objective function to the target and trial images
            targetObjApplied = computeRMSEForColl(timeSeries,targetVectorMultiband);
            trialObjApplied = computeRMSEForColl(timeSeries,trialVectorMultiband);
            finalVectorToStore = targetVectorMultiband.where(trialObjApplied.lt(targetObjApplied),trialVectorMultiband);
            
            return finalVectorToStore;
        
        mutationColl = popsizeColl.map(mutateCollection)
        mutationArray = mutationColl.toBands().toArray().arrayReshape(ee.Image([pop_size,numOfVarsToOpt]).toArray(),2);
        
        # Convert the array to a collection for RMSE calculation
        def convertToArrayForScreeRMSE(i):
            # Acquire the iteration number
            iterNum = ee.Number(ee.Image(i).get('iter_num'));
            iterNumPlus1 = iterNum.add(1);
            # Save the nth array
            mutationNVector = ee.Image(mutationArray).arraySlice(0,iterNum,iterNumPlus1);
            # Flatten to an image
            imageToReturn = mutationNVector.arrayProject([1]).arrayFlatten([varsList]);
            return imageToReturn;
        icForRMSECalc = popsizeColl.map(convertToArrayForScreeRMSE)
        finalCollForScreeCacl = addRMSE(icForRMSECalc,timeSeries);
        
        # Calculate the best/lowest RMSE
        finalRMSEImage = ee.Image(finalCollForScreeCacl.select('RMSE').reduce('min',parallelScale));
        screeBandsToAdd = (ee.Image(ee.Image(iPI).get('scree')).addBands(finalRMSEImage.rename(ee.String('RMSE_').cat(ee.Number.parse(screeBandNum).add(numExtantScreeBandsPlus1).format('%03d')))));
        
        return mutationArray.set('scree',ee.Image(screeBandsToAdd)).set('seed_num',ee.Number(seedFactor).divide(1e12));
    
    # According to the two image arguments, accept pre-defined population and scree images
    if (startingPopulationImage == None) and (existingScreeImage == None):
        screeImageForIterate = ee.Image();
        startingImageForIterate = initialPopImage.set('seed_num', 0).set('scree',screeImageForIterate);
    elif (startingPopulationImage == None) and (existingScreeImage != None):
        screeImageForIterate = ee.Image(existingScreeImage);
        startingImageForIterate = initialPopImage.set('seed_num', 0).set('scree', screeImageForIterate);
    elif (startingPopulationImage != None) and (existingScreeImage == None):
        screeImageForIterate = ee.Image();
        startingImageForIterate = ee.Image(startingPopulationImage).set('scree', screeImageForIterate);
    else:
        screeImageForIterate = ee.Image(existingScreeImage);
        startingImageForIterate = ee.Image(startingPopulationImage).set('scree', screeImageForIterate);
    
    # Iterate the function the specified number of times
    def collectionIterate(current,result):
        return mutatePopulation(result,ee.Number(current.get('seed_num')).multiply(1e12),ee.Number(current.get('system:index')));
    populationOutput = iterColl.iterate(collectionIterate,startingImageForIterate);
    screeImage = ee.Image(ee.Image(populationOutput).get('scree'));
    
    def flattenImageColl(i):
        # Acquire the iteration number
        iterNum = ee.Number(ee.Image(i).get('iter_num'));
        iterNumPlus1 = iterNum.add(1);
        # Save the nth array
        mutationNVector = ee.Image(populationOutput).arraySlice(0,iterNum,iterNumPlus1);
        # Flatten to an image
        imageToReturn = mutationNVector.arrayProject([1]).arrayFlatten([varsList]);
        return imageToReturn;
    flattenedIC = popsizeColl.map(flattenImageColl);
    
    finalColl = addRMSE(flattenedIC,timeSeries);
    
    # Use quality mosaic to determine the best chromosome
    mosaicedImage = finalColl.qualityMosaic('inverseRMSE');
    
    # Return the parameters or the scree plot depending on what is requested
    if (daisyChain == False):
        print("DE Optim is returning a:")
        if (computeScree == True):
            print("Scree Image");
            print("");
            bN = screeImage.bandNames().getInfo()
            rString = re.compile("REMOVE_[0-9]+")
            removeList = list(filter(rString.match, bN))
            return screeImage.select(screeImage.bandNames().removeAll(removeList).removeAll(['REMOVE']));
        else:
            print("Coefficients Image");
            print("");
            return mosaicedImage.select(mosaicedImage.bandNames().remove('inverseRMSE'));
    else:
        print("DE Optim is returning a:")
        if (computeScree == False):
            print("Population Image");
            print("");
            return ee.Image(populationOutput);
        else:
            print("Scree Image");
            print("");
            bN = screeImage.bandNames().getInfo()
            rString = re.compile("REMOVE_[0-9]+")
            removeList = list(filter(rString.match, bN))
            return screeImage.select(screeImage.bandNames().removeAll(removeList));