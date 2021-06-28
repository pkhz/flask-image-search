#Step 4 - Performing Search
#import the necessary packages
from colordescriptor import ColorDescriptor
from searcher import Searcher
import argparse
import cv2

#construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", required = True, help = "Path to where the computed index will be stored")
ap.add_argument("-q", "--query", required = True, help = "Path to the query image")
ap.add_argument("-r", "--result-path", required = True, help = "Path to the result path")
args = vars(ap.parse_args())

#initialize image descriptor
cd = ColorDescriptor((8, 12, 3)) #same val as in index.py

#load query image and describe it
query = cv2.imread(args["query"])
features = cd.describe(query) #extract features

#perform the search
searcher = Searcher(args["index"]) #call searcher.py
results = searcher.search(features)

#display query
cv2.imshow("Query", query)

#loop over results
for (score, resultID) in results:
    #load result image and display it
    print(resultID)
    print(score)
    
    #print(args["result_path"])
    result = cv2.imread(resultID)
    #print(result)
    cv2.imshow("Result", result)
    cv2.waitKey(0)