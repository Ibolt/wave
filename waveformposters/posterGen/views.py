from django.shortcuts import render
from django.http import HttpResponse

from .waveformGen import search, audioDownload, getMainColours, graphSound

# Create your views here.
def index(request):
	# Put a placeholder image here maybe
	waveform_meta = ["", "", "", "", "", ""]
	query = ""

	if request.method == "GET":
		try:
			query = request.GET['user_query']
		except:
			query = ""

		if not (query == ""):
			searchResults = search(query)

			# try:
			audioDownload(searchResults[0])
			colours = getMainColours(searchResults[1])
			waveform_meta  = graphSound(colours[1], colours[0], searchResults[3], searchResults[2])
			waveform_meta[5] = searchResults[0]

			# except Exception as e:
			#print("Could not download audio file. \n")

	return render(request, 'posterGen/index.html', context={'waveform_meta': waveform_meta})