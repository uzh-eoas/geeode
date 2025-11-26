#!/usr/bin/env python

# Native packages
import time
import datetime
import sys
import re

# Required packages
import ee
import pandas as pd


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Analytical Functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def de_optim(pop_size = int,
             iNum = int ,
             funToOpt = str,
             inputVars = list[str],
             inputBounds = list[list[float]],
             timeSeries = ee.ImageCollection,
             bandName = str,
             optParams = {}):
    
    """
    Perform differential‑evolution optimization on an Earth Engine time‑series.

    Args:
        pop_size (int): the number of population members when generating a new population
        iNum (int): the number of iterations
        funToOpt (str): the function you want to optimize in the form of a GEE expression; the function must include the time band as well as all variables you'll be optimizing; band names must be formatted using the prescribed GEE format `b('...')` 
        inputVars (list[str]): a list of the names of the variables you will optimize
        inputBounds (list[list[float]]): a list of lists; each sublist describes the numeric bounds (inclusive) that will be used for each variable; the order must match the `inputVars` list
        timeSeries (ee.ImageCollection): the image collection time series being modelled; each image must have the raw data being modelled as a band name (identified below) as well as a time band and a properly formatted `system:footprint`
        bandName (str): the band name of the raw data being modelled; each image in the time series must contain this band
        optParams (dict, optional): optional dictionary of additional parameters; see "Other Parameters" below
    
    Other Parameters:
        timeBand (String, default 'time'): name of the time band:
        mutationStrategy (String, default rand): mutation strategy; possible values are `rand` and `best`
        M (Float, default 0.5): mutation factor $M in (0,2]$ within the function $v_x + M * (v_y – v_z)$ where $v_x$,$v_y$, and $v_z$ are population members
        cr (Float, default 0.7): crossover factor for the binomial selection process $cr in (0,1)$
        computeScree (Boolean, default False): `True` if you want to return a scree image, else `False`
        initialPopSeed (Integer, default 1): the seed used to generate the initial population
        parallelScale (Integer, default 1): the `parallelScale` value used to input into relevant EE functions; $in {1 ... 16}$
        daisyChain (Boolean, default False): `True` if you want to return the final iteration population image (i.e., an array-image), else `False`
        startingPopulationImage (ee.Image, default null): the `ee.Image` object to input as a starting population image; must have a Integer value property `seedNum` and must be an array-image formatted according to the variables being optimized
        existingScreeImage (ee.Image, default null): the `ee.Image` image to input as a starting scree image; should be an object computed from `computeScree`
        verbosePrinting (Boolean, default False): if `True`, print info upong running
        

    Returns:
        (ee.Image): a coefficients image, scree‑plot image, or population image depending on the options selected (e.g., ``computeScree`` or ``daisyChain``)
        
    Raises:
        ValueError: if required inputs are missing or malformed (e.g., mismatched variable lists, missing bounds, or invalid mutation strategy)

    Notes:
        * The function performs extensive type‑checking and default‑value handling before running the core DE algorithm.
        * It supports optional “daisy‑chain” processing and can generate a scree plot of RMSE values across iterations.

    Examples:
        ```python
        result = de_optim(
            pop_size=10,
            iNum=25,
            funToOpt="b('time') * b('a') + b('b')",
            inputVars=["a", "b"],
            inputBounds=[[0, 1], [0, 10]],
            timeSeries=time_series,
            bandName="NDVI",
        )
        ```
             
    """
    
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
    
    try:
        verbosePrinting = optParams['verbosePrinting']
    except KeyError:
        optParams['verbosePrinting'] = False
        verbosePrinting = optParams['verbosePrinting']

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
        # Map a function across the inputted collection to make a modeled layer,
        # then sum the residuals and compute RMSE
        def makeSquaredResiduals(i):
            bandsAdded = i.addBands(imageToAssess);
            modeledValue = bandsAdded.addBands(bandsAdded.expression(funToOpt).rename('modeled'));
            return modeledValue.addBands(modeledValue.expression(desiredString).rename('squaredResiduals'));
        modeledValue = collToUse.map(makeSquaredResiduals);
        meanRMSE = ee.Image(modeledValue.select('squaredResiduals').reduce('mean',parallelScale)).rename('mean');
        finalRMSE = meanRMSE.sqrt().rename('RMSE');
        return finalRMSE;
    
    # Add RMSE to each set of potential coefficients using the time series
    def addRMSE(collWithCoeffs,timeSeries):
        # Map a function across the inputted collection to make a modeled layer,
        # then sum the residuals and compute RMSE
        def makeRMSEImages(i):
            def makeSquaredResiduals(tsi):
                tsWithBands = tsi.addBands(i);
                tsWithPredictions = tsWithBands.addBands(tsWithBands.expression(funToOpt).rename('modeled'));
                tsWithSquaredResids = i.addBands(tsWithPredictions.expression(desiredString).rename('squaredResiduals'));
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
        if verbosePrinting == True:
            print("DE Optim is returning a:")
        if (computeScree == True):
            if verbosePrinting == True:
                print("Scree Image");
                print("");
            bN = screeImage.bandNames().getInfo()
            rString = re.compile("REMOVE_[0-9]+")
            removeList = list(filter(rString.match, bN))
            return screeImage.select(screeImage.bandNames().removeAll(removeList).removeAll(['REMOVE']));
        else:
            if verbosePrinting == True:
                print("Coefficients Image");
                print("");
            return mosaicedImage.select(mosaicedImage.bandNames().remove('inverseRMSE'));
    else:
        if verbosePrinting == True:
            print("DE Optim is returning a:")
        if (computeScree == False):
            if verbosePrinting == True:
                print("Population Image");
                print("");
            return ee.Image(populationOutput);
        else:
            if verbosePrinting == True:
                print("Scree Image");
                print("");
            bN = screeImage.bandNames().getInfo()
            rString = re.compile("REMOVE_[0-9]+")
            removeList = list(filter(rString.match, bN))
            return screeImage.select(screeImage.bandNames().removeAll(removeList));


def sub_sample(iC = ee.ImageCollection,
               nKeep = int,
               sType = str,
               bandName = str,
               optParams = {}):
    """
    Subsamples an image collection by temporal density.

    Args:
        iC (ee.ImageCollection): the GEE image collection to subsample
        nKeep (int): the number of observations to keep at every pixel (maximum)
        sType (str): the subsampling type you'd like to perform; one of 'bulk', 'splitshuffle', leapfrog'
        bandName (str): the name of the band (in the image collection) containing your value of interest
    
    Other Parameters:  
        nStD (float, default 0.5): the number of standard deviations to use a kernel width when calculating temporal density
        timeBandName (str, default 'time'): the name of time band in each image
        sN (int, default 4): the number of splits if using the 'splitshuffle' mtehod
        seedNum (int, default 1): the random seed use for shuffling
        verbosePrinting (Boolean, default False): if `True`, print info upon running

    Returns:
        (ee.Image): an image comprised of `nKeep` paired bands; each pair of bands includes the time value and the original observed band values at that time
    
    Raises:
        ValueError: if required inputs are missing or malformed (e.g., `nStD` must be positive)

    Notes:
        * Time data from the individual images in the input collection will be lost after using this function.

    Examples:
        ```python
        sampled_collection = sub_sample(
            iC=your_image_collection,
            nKeep=30,
            sType='leapfrog',
            bandName='NDVI
        )
        ```
    
    """
    
    # Set optional values to defaults if they are not explicitly defined
    try:
        nStD = optParams['nStD']
    except KeyError:
        optParams['nStD'] = 0.5
        nStD = optParams['nStD']
    
    if nStD <= 0:
        raise ValueError('nStD must be greater than 0!')

    try:
        timeBandName = optParams['timeBandName']
    except KeyError:
        optParams['timeBandName'] = 'time'
        timeBandName = optParams['timeBandName']
    
    try:
        sN = optParams['sN']
    except KeyError:
        optParams['sN'] = 4
        sN = optParams['sN']
    
    if sN <= 0:
        raise ValueError('sN must be greater than 0!')
    
    try:
        seedNum = optParams['seedNum']
    except KeyError:
        optParams['seedNum'] = 1
        seedNum = optParams['seedNum']
    
    try:
        verbosePrinting = optParams['verbosePrinting']
    except KeyError:
        optParams['verbosePrinting'] = False
        verbosePrinting = optParams['verbosePrinting']
    
    # Calculate the desired time standard deviation value
    i_c_time_std_dev_nstd = iC.select(timeBandName).reduce(ee.Reducer.stdDev()).multiply(nStD)

    # Add ± min/max times (based on nStD) for temporal filtering
    def map_i(i):
        return i.addBands(i.select(timeBandName).add(i_c_time_std_dev_nstd).rename('max_time')) \
                 .addBands(i.select(timeBandName).subtract(i_c_time_std_dev_nstd).rename('minTime'))

    i_c_with_min_max = iC.map(map_i)

    # Filter based on ±min/max
    def min_max_filter(o):
        def min_max_inner(i):
            test_band = i.select(timeBandName).gt(o.select('minTime')) \
                        .And(i.select(timeBandName).lt(o.select('max_time'))) \
                        .rename('density').updateMask(i.select(bandName).mask())
            value_to_sum = i.addBands(test_band)
            return value_to_sum

        density_band = i_c_with_min_max.map(min_max_inner).select('density').sum()
        return o.addBands(density_band.updateMask(o.select(bandName).mask())) \
                 .addBands(ee.Image.random(seedNum).add(1).rename('random').updateMask(o.select(bandName).mask())) \
                 .addBands(density_band.multiply(ee.Image.random(seedNum).add(1)).rename('weight').updateMask(o.select(bandName).mask()))

    density_coll = i_c_with_min_max.map(min_max_filter)

    keys_shuffle = density_coll.select('weight').toArray()

    # Shuffle the time series values according to temporal density weight
    original_ts = density_coll.select(bandName, timeBandName).toArray()
    density_coll_shuffled = original_ts.arraySort(keys_shuffle)

    # Format array images that will serve as array masks, allowing for subsampled values to 
    # be returned or for their complement to be returned (i.e., the masked out values)
    ts_length = density_coll_shuffled.arrayLength(0)
    n_to_remove_diff = ee.Image(ts_length).subtract(nKeep)
    n_to_remove = n_to_remove_diff.where(n_to_remove_diff.lte(ee.Image.constant(0)), 0)
    on_off_array = ee.Image([1, 0]).toArray()
    off_off_array = ee.Image([0, 0]).toArray()
    on_off_repeated = on_off_array.arrayRepeat(0, n_to_remove)
    on_off_repeated_sliced = on_off_repeated.arraySlice(0, 0, ts_length)
    on_off_repeated_sliced_length = on_off_repeated_sliced.arrayLength(0)
    on_off_repeated_keep_sliced_keep_sum = on_off_repeated_sliced.arrayReduce('sum', [0])
    additional_number_to_remove = on_off_repeated_keep_sliced_keep_sum.multiply(ee.Image([0])) \
        .where(on_off_repeated_keep_sliced_keep_sum.arrayFlatten(
            [['n']]
        ).subtract(ee.Image(nKeep)).gt(0),
            on_off_repeated_keep_sliced_keep_sum.subtract(nKeep))
    on_off_repeated_length = on_off_repeated.arrayLength(0)
    off_off_repeated = off_off_array.arrayRepeat(0, additional_number_to_remove.arrayFlatten(
        [['n']]
    ))
    off_off_repeated_length = off_off_repeated.arrayLength(0)

    # Format arrays that will be used if the default leap-frog sampling "laps" the
    # total number of samples (i.e., the sampling removes more than half of the original time series)
    aggro_slice_length = on_off_repeated_sliced_length.subtract(off_off_repeated_length)
    on_off_repeated_sliced_with_aggro_sampling = on_off_repeated_sliced.arraySlice(0, 0, aggro_slice_length)
    aggro_mask_array = on_off_repeated_sliced_with_aggro_sampling.arrayCat(off_off_repeated, 0)
    keep_vector = ee.Image([1]).arrayRepeat(0, ts_length.subtract(on_off_repeated_length))
    normal_mask_array = keep_vector.arrayCat(on_off_repeated, 0)

    # Finalize the mask to use for the time series
    mask_test = on_off_repeated_length.gt(ts_length)
    mask_array = normal_mask_array.where(mask_test, aggro_mask_array)

    # Format the band names for the flattened image
    def format_number(n):
        return ee.String('b').cat(ee.Number(n).format('%02d'))

    n_l = ee.List.sequence(1, nKeep).map(format_number)

    # Bulk sampling
    bulk_sampling_sliced = density_coll_shuffled.arraySlice(0, 0, nKeep)
    bulk_sampling_time_keys = bulk_sampling_sliced.arraySlice(1, -1)
    bulk_sampled_array_image = bulk_sampling_sliced.arraySort(bulk_sampling_time_keys)
    sub_sampled_image_bulk = bulk_sampled_array_image.arrayPad([nKeep, 2]).arrayFlatten([n_l, [bandName, timeBandName]], '_').selfMask()

    # Split shuffle sampling
    # Use the sN (number of splits) input to split the array pseudo randomly
    # (i.e., weighted by temporal density) and subsample them
    def make_constant_image(i):
        return ee.Image.constant(i).set('n', i)

    n_coll = ee.ImageCollection(ee.List.sequence(0, (sN - 1)).map(make_constant_image))

    def array_slice_collection(i):
        return ee.Image(density_coll_shuffled).arraySlice(axis = 0, start = ee.Image(i).int(), step = sN)

    split_shuffled_arrays = n_coll.map(array_slice_collection).toArrayPerBand()
    density_coll_sliced_split_shuffled = split_shuffled_arrays.arraySlice(axis = 0, end = ee.Image.constant(nKeep))
    keys_time_split_shuffled = density_coll_sliced_split_shuffled.arraySlice(1, -1)
    density_coll_sorted_split_shuffled = density_coll_sliced_split_shuffled.arraySort(keys_time_split_shuffled)
    density_coll_padded_split_shuffled = density_coll_sorted_split_shuffled.arrayPad([nKeep, 2])
    sub_sampled_image_split_shuffled = density_coll_padded_split_shuffled.arrayFlatten([n_l, [bandName, timeBandName]], '_').selfMask()

    # Apply the array-mask to confirm the subsampling (leapfrog)
    density_coll_masked_lf = density_coll_shuffled.arrayMask(mask_array.arrayRepeat(1, ee.Image(1)))
    time_sorting_keys_lf = density_coll_masked_lf.arraySlice(axis = 1, start = ee.Image.constant(-1))
    density_coll_masked_sorted_lf = density_coll_masked_lf.arraySort(time_sorting_keys_lf)

    # Mask the flattened array-image with itself to remove 0 values
    sub_sampled_image_lf = density_coll_masked_sorted_lf.arrayPad([nKeep, 2]).arrayFlatten([n_l, [bandName, timeBandName]], '_').selfMask()

    # Return the image collection of interest
    if sType == 'bulk':
        if verbosePrinting == True:
            print('Bulk Sampling')
        image_to_return = sub_sampled_image_bulk
    elif sType == 'splitshuffle':
        if verbosePrinting == True:
            print('Split Shuffle Sampling')
        image_to_return = sub_sampled_image_split_shuffled
    elif sType == 'leapfrog':
        if verbosePrinting == True:
            print('Leapfrog Sampling')
        image_to_return = sub_sampled_image_lf
    else:
        raise ValueError("Input one of: 'bulk', 'splitshuffle', or 'leapfrog'.")
        image_to_return = None

    return image_to_return


def ts_image_to_coll(ts_image = ee.Image,
                     band_name = str,
                     ts_length = int,):
    """

    A function used to take an outputted image from the `sub_sample` function
    and transforms it back into an image collection wherein every image has a 'time'
    band and a band the contains the original sampled observation at that time.

    Images in the collection have no time data at the asset/image level; time data
    now exists as a band value. The first image in the collection has the earliest
    observation for each pixel (subsampled from the original stack of values at each
    pixel), the second image has the second earliest, and so on in temporal order.

    Args:
     ts_image (ee.Image): the image outputted from the `sub_sample` function
     band_name (str): the original band name of the values being sampled
     ts_length (int): the number of samples maintained from the original time series per pixel

    Returns:
    (ee.ImageCollection): returns an image collection with the time and original observations as bands in each image
    
    Examples:
        ```python
        sampled_collection = sub_sample(
            i_c=your_image_collection,
            n_keep=n_observations_to_keep,
            s_type='leapfrog',
            band_name='NDVI'
        )
        ts_image_to_coll(sampled_collection,'NDVI',n_observations_to_keep)
        ```
    """
    # Make a function to create a client-side list of numbers
    def make_iter_array(start, end):
        return list(range(start, end + 1))

    # Make a function to pad the number with leading zeros
    def zero_pad(num, places):
        return str(num).zfill(places)

    # Make a function that cracks open the image bands and returns them in a form that the ee.ImageCollection.fromImages() function can handle
    def unzip_bands(i):
        padded_num = zero_pad(i + 1, 2)
        main_band = f"b{padded_num}_{band_name}"
        time_band = f"b{padded_num}_time"

        image_to_add = ee.Image(ts_image).select([main_band, time_band], [{band_name}, 'time'])
        return image_to_add

    # Create the list of indices
    i_l = make_iter_array(0, ts_length - 1)

    # Create the ImageCollection
    ts_coll = ee.ImageCollection.fromImages([unzip_bands(i) for i in i_l])
    return ts_coll


def apply_model(time_series = ee.ImageCollection,
                de_optim_output = ee.Image,
                fun_to_opt = str):
    """
    Apply a model expression to each image in a collection.

    Args:
        time_series (ee.ImageCollection): The input time series ImageCollection on which to apply the model
        de_optim_output (ee.Image): The coefficient image returned by `de_optim`
        fun_to_opt (str): A string expression to evaluate; e.g., "b('time') * b('a') + b('b')"

    Returns:
        (ee.ImageCollection): The collection with a new band named 'predicted'
    """
    
    def apply_expression(image):
        # Add the de_optim_output bands, evaluate the expression, and rename the result.
        predicted = image.addBands(de_optim_output) \
                       .expression(fun_to_opt) \
                       .rename('predicted')
        # Preserve original bands.
        return image.addBands(predicted)

    coll_with_predictions = time_series.map(apply_expression)
    return coll_with_predictions


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ## Helper Functions ##
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def check_for_tasks(unique_string = str):
    """

    A helper function used to check if any tasks within the Earth Engine task queue
    contains `unique_string` within their description and are currently in a RUNNING
    or READY state.

    Args:
     unique_string (str): will be used to find tasks in your queue that are RUNNING or READY (matched against each tasks `description`)
    
    Returns:
        (bool): `True` if at least one matching task is found; otherwise returns `False`
    
    Examples:
        ```python
        if test_for_tasks('task_unique_string') == True:
            <do_something_specific_if_tasks_are_running>
        ```
    """
    
    # Get all task descriptions as strings
    task_list = [str(t) for t in ee.batch.Task.list()]

    # Keep only those that contain the unique identifier
    subset_list = [s for s in task_list if unique_string in s]

    # Further filter for tasks that are RUNNING or READY
    active_tasks = [
        s for s in subset_list
        if any(state in s for state in ('RUNNING', 'READY'))
    ]

    # Return True if there is at least one such task
    return len(active_tasks) > 0


def check_for_asset_then_run_task(asset_id_to_test = str,
                                 task_to_start = ee.batch.Task,
                                 unique_string = str):
    """
    A helper function used to check if an asset already exists (or a relevant task
    is running); if not, then it starts a task of interest

    Args:
        asset_id_to_test (str): asset id that you will test
        task_to_start (ee.batch.Task): the task you would like to start
        unique_string (str): a unique string used to test if the task is already running

    Notes:
        * It's best to align this function with specific variables used within your
          larger workflow; see the example below.
        * Specifically consider using a job `description` as the `unique_string`

    Examples:
        ```python
        your_task = ee.batch.Export.<image_or_table>.toAsset(
                                             ...<other_arguments>...
                                             assetId = asset_id_to_test,
                                             description = unique_string,
        )
        test_for_asset_then_run_task(asset_id_to_test,your_task,unique_string)

        ```

    """
    try:
        # Try to see if the asset exists then except errors or proceed accordingly.
        ee.data.getAsset(asset_id_to_test)
        print(f'Asset already exists: {asset_id_to_test}\n')
    except ee.EEException:
        if check_for_tasks(unique_string):
            print(f'Asset task is already queued: {asset_id_to_test}\n')
        else:
            try:
                # Try to start the task and return and error if something goes wrong
                task_to_start.start()
                print(f'Asset task started: {asset_id_to_test}\n')
            except Exception as error:
                print(f'Something went wrong when starting the task: {asset_id_to_test}\n')
    # Return any overarching error if something goes wrong with checking the asset
    except Exception as error:
        print(f'Something went wrong when checking for the asset: {asset_id_to_test}')
        print(error)
        print('')


def pause_and_wait(unique_id = str,
                   wait_time = 60,
                   try_again = False,
                   max_time = None):
    """
    A helper function used to take a "pause" in a workflow to "wait" for tasks to finish.
    
    Every time the function runs, it checks for tasks in the queue that contain 'unique'
    as a substring in their description and are queued either as RUNNING or READY. The
    function then lists the number of tasks (1 or more) that match the criteria, prints
    how many there are alongside the time, then waits the specified number of checks
    before following the same protocol once again.

    Args:
        unique_id (str): will be used to find tasks in your queue that are RUNNING or READY
        wait_time (int): the number of seconds to wait before rechecking for tasks
        try_again (bool): if True and there are errors, continue trying to check for tasks
        max_time (int): tasks
    
    Raises:
        ValueError: If required inputs are malformed (e.g., `wait_time` must be positive).

    Notes:
        * To interrupt the waiting, simply use a keyboard escape method.
        * Use the `try_again` and `max_time` arguments deliberately.
            * `try_again` can cause the code to continue indefinitely
            * `max_time` can/will CANCEL tasks in the queue

    Examples:
        ```python
        pauseAndWait('task_unique_string')
        ```
    
    """

    # Raise an error if the wait time is not a positive integer value
    if wait_time <= 0:
        raise ValueError('You must specify the wait time as a positive integer (i.e., number of seconds).')
    
    while True:
        try:
            taskList = [str(i) for i in ee.batch.Task.list()]
            subsetList = [s for s in taskList if unique_id in s]
            subsubList = [s for s in subsetList if any(xs in s for xs in ['RUNNING', 'READY'])]
            count = len(subsubList)
            if count == 0:
                print('No jobs running!\n')
            else:
                count = 1
                while count >= 1:
                    if max_time != None and type(max_time) == int and max_time > 0:
                        rawTaskList = ee.data.listOperations()
                        taskList = pd.json_normalize(rawTaskList)
                        running = taskList.loc[taskList.loc[:,'metadata.state'] == 'RUNNING']
                        running_copy = running.copy()
                        running_copy.loc[:,'startSSE'] = running_copy.apply(lambda row: time.mktime(time.strptime(row['metadata.startTime'].replace("-","/"),"%Y/%m/%dT%H:%M:%S.%fZ")),axis=1)
                        running_copy.loc[:,'currentSSE'] = running_copy.apply(lambda row: time.mktime(time.strptime(row['metadata.updateTime'].replace("-","/"),"%Y/%m/%dT%H:%M:%S.%fZ")),axis=1)
                        running_copy.loc[:,'duration'] = running_copy.loc[:,'currentSSE'] - running_copy.loc[:,'startSSE']
                        opNamesToCancel = running[running_copy.loc[:,'duration'] > max_time]
                        jobsToCancel = [str(s).split('/')[-1] for s in opNamesToCancel['name']]
                        for task in rawTaskList:
                            for job_id in jobsToCancel:
                                if job_id in task["name"]:
                                    print("Canceling task: ",job_id)
                                    ee.data.cancelOperation(task["name"])
                                    print("\n")
                    taskList = [str(i) for i in ee.batch.Task.list()]
                    subsetList = [s for s in taskList if unique_id in s]
                    subsubList = [s for s in subsetList if any(xs in s for xs in ['RUNNING', 'READY'])]
                    count = len(subsubList)
                    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), 'Number of queued tasks:', count)
                    time.sleep(wait_time)
            print('Moving on...\n')
        except (KeyboardInterrupt):
            sys.exit('Pipeline stopped via keyboard interrupt.')
        except Exception as error:
            print("Something went wrong when checking the tasks:")
            print(error)
            print("\n")
            if try_again == True:
                print("Trying again...\n")
                time.sleep(5)
                continue
        break


