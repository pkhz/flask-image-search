#import the necessary packages
import numpy as np
import csv

class Searcher:
    def __init__(self, indexPath):
        #store our index path -  path to where our index.csv
        self.indexPath = indexPath

    #limit - max no of results
    def search(self, queryFeatures, limit =7):
        #initialize our dictionary of results
        results = {}

        #open the index file for reading
        with open(self.indexPath) as f:
            #initalize CSV reader
            reader = csv.reader(f)

            #loop over the rows in the index
            for row in reader:
                #parse out image ID and features, then compute the chi-squared distance between features in our index and query features
                features = [float(x) for x in row[1:]]
                d = self.chi2_distance(features, queryFeatures) #distance between two feature vectors

                #update results dictionary - key=current image ID, value=distance(hows similar both image is)
                results[row[0]] = d
            
            #close the reader
            f.close()
        
        #sort our results, so that the smaller distances(i.e. the most relevant images) are at front of list
        results = sorted([(v,k) for (k,v) in results.items()])

        #return results
        return results[:limit]
    
    def chi2_distance(self, histA, histB, eps = 1e-10):
        #compute the chi-squared distance
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
            for (a, b) in zip(histA, histB)])
        
        #return the chi-squared distance
        return d