import os
import numpy as np

from flask import Flask, render_template, request, jsonify

from pyimagesearch.colordescriptor import ColorDescriptor
from pyimagesearch.searcher import Searcher

#create flask instance
app = Flask(__name__, static_url_path = "", static_folder = "static")
#app = Flask(__name__)

INDEX = os.path.join(os.path.dirname(__file__), 'index.csv')

#main route
@app.route('/')
def index():
    return render_template('index.html') #return index.html

#search route
@app.route('/search', methods=['POST'])
def search():

    if request.method == "POST":

        RESULTS_ARRAY = []

        #get url
        image_url = request.form.get('img')

        try:

            #initialize the image descriptor
            cd = ColorDescriptor((8, 12, 3))
            
            #load the query image and describe it
            from skimage import io
            import cv2
            print(image_url)
            image_url2 = "static/"
            image_url = image_url2 + image_url
            print(image_url)
            #query =  io.imread(image_url)
            #query = (query * 255).astype("uint8")
            #(r, g, b) = cv2.split(query)
            #query = (cv2.merge([r, g, b]))
            #query = query[::-1]
            query = cv2.imread(image_url)
            features = cd.describe(query)

            #perform the search
            searcher = Searcher(INDEX)
            results = searcher.search(features)

            #loop over the results, displaying the score and image name
            for (score, resultID) in results:
                print (resultID)
                print (score)
                RESULTS_ARRAY.append({"image": str(resultID), "score": str(score)})
            
            RESULTS_ARRAY.pop(0)

            #return success
            return jsonify(results=(RESULTS_ARRAY[::1][:3])) #sort asc, top 3 only
        
        except:
            #return error
            return jsonify({"sorry":"Sorry, no results! Please try again."}), 500

# run!
if __name__ == '__main__':
    app.run('0.0.0.0', debug=True)
