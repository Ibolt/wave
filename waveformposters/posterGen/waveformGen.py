import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pytube import YouTube
import glob
import os
from PIL import Image
from io import BytesIO
from urllib import request
import cv2
from sklearn.cluster import KMeans
from youtube_search import YoutubeSearch
from uuid import uuid4
import matplotlib.colors as colours

'''
NEXT STEPS
******************

-	Add search functionality, try using youtube api first, then third party apis, then maybe urllib/requests
-	Potential issue with newer videos/videos with less views
- 	Potentially use lower-res image from yt search to generate colours for better performance
-	Compare perfomance of getting metadata from yt search vs pytube
-	Compare colour conversion performance
'''
	
#Get audio from YouTube
def audioDownload(link):
	yt = YouTube(link)

	yt.streams.filter(only_audio=True)[0].download("C:/Users/Imaan/Documents/Programs/WaveformPostersSite/waveformposters/posterGen/static/audio")

	# info = [yt.title, yt.thumbnail_url, yt.author]
	# return info

#Create histogram for pixels of image
def make_histogram(cluster):
    """
    Count the number of pixels in each cluster
    :param: KMeans cluster
    :return: numpy histogram
    """
    numLabels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    hist, _ = np.histogram(cluster.labels_, bins=numLabels)
    hist = hist.astype('float32')
    hist /= hist.sum()
    return hist

#Determine the primary and secondary colour of the thumbnail of the song
def getMainColours(imageUrl):
	#Download the image from the thumbnail url
	request.urlretrieve(imageUrl, "C:/Users/Imaan/Documents/Programs/WaveformPostersSite/waveformposters/posterGen/static/thumbnail.jpg")
	
	img = cv2.imread("C:/Users/Imaan/Documents/Programs/WaveformPostersSite/waveformposters/posterGen/static/thumbnail.jpg")

	#Delete the image once it's been downloaded
	os.remove("C:/Users/Imaan/Documents/Programs/WaveformPostersSite/waveformposters/posterGen/static/thumbnail.jpg")

	height, width, _ = np.shape(img)

	# reshape the image to be a simple list of RGB pixels
	image = img.reshape((height * width, 3))

	# we'll pick the 5 most common colors
	num_clusters = 5
	clusters = KMeans(n_clusters=num_clusters)
	clusters.fit(image)

	# count the dominant colors and put them in groups
	histogram = make_histogram(clusters)

	# then sort them, most-common first
	combined = zip(histogram, clusters.cluster_centers_)
	combined = sorted(combined, key=lambda x: x[0], reverse=True)

	finalColours = [[], []]

	for i in range(2):
		for x in range(3):
			finalColours[i].append(round(combined[i][1][x]) / 255)

		#Colours aren't output in rgb so rearrange the list
		finalColours[i] = finalColours[i][::-1]

	return finalColours

#Graph audio
def graphSound(colour, bg, title, artist):

	#Get all files in folder
	allFiles = glob.glob("C:/Users/Imaan/Documents/Programs/WaveformPostersSite/waveformposters/posterGen/static/audio/*")
	audio_path = max(allFiles, key=os.path.getctime)

	#Load audio file, set a sample rate of 44100Hz
	x,sr = librosa.load(audio_path)

	#Delete the file once the data has been taken from it
	os.remove(audio_path)

	#Size of the figure in inches
	fig = plt.figure(figsize=(9.6,5.4))

	#Set up the axes object
	ax = fig.add_axes([0.25,0.20,0.5,0.5])

	#Make the axes transparent and remove the ticks
	ax.spines['bottom'].set_color((1.0,1.0,1.0,0.0))
	ax.spines['left'].set_color((1.0,1.0,1.0,0.0))
	ax.spines['top'].set_color((1.0,1.0,1.0,0.0))
	ax.spines['right'].set_color((1.0,1.0,1.0,0.0))

	ax.tick_params(axis='x', colors=(1.0,1.0,1.0,0.0))
	ax.tick_params(axis='y', colors=(1.0,1.0,1.0,0.0))

	#Generate the waveform using librosa and matplotlib
	librosa.display.waveplot(x, sr=sr, ax=ax, color=colour)

	#Determine if white or black text should be used based on the background colour
	if ((bg[0] * 255 * 0.288) + (bg[1] * 255 * 0.587) + (bg[2] * 255 * 0.114)) > 186:
		fontColour = "black"
	else:
		fontColour = "white"

	#Font for text
	font = {'fontname': 'Bebas Neue'}

	#Set the title/subtitle of the image
	mpl.rcParams['text.color'] = colour
	mpl.rcParams['axes.labelcolor'] = colour

	plt.title(title, **font, fontsize=28, color=fontColour)
	plt.xlabel(artist, **font, fontsize=20, color=fontColour)

	fig.set_facecolor(bg)
	ax.set_facecolor(bg)

	#Save the figure
	filename = "/static/" + str(uuid4()) + ".png"
	plt.savefig("C:/Users/Imaan/Documents/Programs/WaveformPostersSite/waveformposters/posterGen/" + filename, facecolor=bg)
	#plt.show()
	return [filename, colours.rgb2hex(colour), colurs.rgb2hex(bg), title, artist]

# def decimalToHex(rgbDecimalList):
# 	rgbList = list(map(lambda x: int(x * 255), rgbDecimalList))
# 	return '#%02x%02x%02x' % (rgbList[0], rgbList[1], rgbList[2])
# 	return colours.rgb2hex(rgbDecimalList)

#Search YouTube for the given query, return video link, thumbnail, channel name, and video title
def search(query):
	results = YoutubeSearch(query, max_results=1).to_dict()[0]

	return ["https://youtu.be/" + results["id"], results["thumbnails"][0], results["channel"], results["title"]]

# searchResults = search("triple double")

#Download the audio file and store key information to a list
# audioDownload(searchResults[0])

# audio_path = "C:/Users/Imaan/Documents/Programs/Waveform Pos'ters/Kendrick Lamar - ELEMENT.mp4"
# colour = '#ffffff'
# bg = '#17202A'

# colours = getMainColours(searchResults[1])
	
# graphSound(colours[1], colours[0], searchResults[3], searchResults[2])