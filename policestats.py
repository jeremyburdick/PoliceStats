import bs4
import pandas as pd
import requests
import re
import numpy as np
import os
import country_converter as coco
import pickle

import statsmodels.api as sm

import math
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea

import plotnine as p9
from plotnine import ggplot, geom_point, aes, stat_smooth, facet_wrap
from locale import setlocale, LC_NUMERIC, atof

from pandas.plotting import scatter_matrix
from matplotlib import pyplot

from adjustText import adjust_text


# cc = coco.CountryConverter()
# cc.data.to_csv('coco.csv')

#exit(0)

FLAG_DIR = 'C:/workflow/policestats/197373-countrys-flags/png/'

setlocale(LC_NUMERIC, 'English_Canada.1252')

def first(x):
	return next(iter(x))

def applyToCol(df, colname, func):
	df[colname] = df[colname].apply(func)

def merge(df1, df2, how='outer'):
	return df1.merge(df2, left_index=True, right_index=True, how=how)

def mergemulti(dfs):
	df = dfs[0]
	for x in enumerate(dfs):
		if x[0] > 0: df = merge(df, x[1])

	return df

def patchCountries(df, patches=None):
	
	if False: 
		if patches is None: patches = {
			'F.S. Micronesia':'Micronesia',
			'Micronesia, Federated States of':'Micronesia',

			'Holy See':'Vatican City',

			'Congo, Republic of the':'Congo',
			'Congo Brazzaville':'Congo',

			'Democratic Republic of the Congo':'DR Congo',
			'Congo, Democratic Republic of the':'DR Congo', 
			'Congo Kinshasa':'DR Congo',

			'The Gambia':'Gambia',
			"People's Republic of China":'China',
			'Bahamas, The':'Bahamas'
			}

		applyToCol(df, 'country', lambda x: patches.get(x, x))
	else:
		cc: coco.CountryCoverter = coco.CountryConverter()

		countries = list(df.country)
		conv = cc.convert(countries, to = 'name_short', not_found = None, src = 'regex')
		df.country = conv
#		cc.convert()

		
		

	return df

def setCountryIndex(df):
	return df.set_index('country')

def patchCountriesAndIndex(df):

	df.country = df.country.apply(lambda x: str(x).strip())

	df = patchCountries(df)

	df = setCountryIndex(df)

	return df



def toFloat(v):
	if type(v) is float:
		return v
	elif type(v) is int:
		return float(v)
	elif v=='–' or v=='-':
		return math.nan
	elif type(v) is str:
		return atof(v)
	else:
		return math.nan

def colToFloat(df, colname):
	applyToCol(df, colname, lambda x: toFloat(x))

def fromURLCacheOrReqGet(id, url):
	file = './_cache_url_' + id + '.html'
	if not os.path.exists(file):
		page = requests.get(url)
		open(file, 'wb').write(page.content)
		return page.content
	else:
		return open(file, 'rb').read()

def soupFromCacheOrReqGet(id, url):
	return bs4.BeautifulSoup(fromCacheOrReqGet(id, url), 'html.parser')

def retrieveStats(config):

	# retrieves url from web or cache
	# passes a soup parser to func
	# excepts a dataframe result from stat with country as index and stat as one of the dataframe variables
	# caches dataframe result to pickle

	mainStat = first(config['stats'].keys())

	statcache = './_cache_df_' + mainStat + '.pickle'

	if os.path.exists(statcache):
		return  pd.read_pickle(statcache)


	soup = bs4.BeautifulSoup(fromURLCacheOrReqGet(mainStat, config['url']), 'html.parser')

	find = config.get('find', None)
	if find is None:
		tabs = soup.findAll('table')
	else:
		tabs = soup.find(find)

	tabIdx = config.get('table', 0)
	df = pd.read_html(str(tabs[tabIdx]))[0]

	country = config.get('country', 0)

	keepOrig = first(config['stats'].values()) is None
	if keepOrig:
		# keep all columns with original names, except index
		statNames = list(df.columns[1:])
		df.columns = ['country'] + statNames
	else:
		colNames = ['country'] + list(config['stats'].keys())
		colIdx = [country] + list(config['stats'].values())

		statNames = config['stats'].keys()

		df = df.iloc[ :, colIdx]
		df.columns = colNames
	

	def custFunc(funcname, df):
		func = config.get(funcname, None)
		if not func is None:
			df = func(df)
		return df

	df = custFunc('drop', df)
	df = custFunc('clean', df)

	for stat in statNames:
		colToFloat(df, stat)

	df = custFunc('calc', df)

	df = patchCountriesAndIndex(df)

	print(df)

	df.to_pickle(statcache)


	return df

def cleanGunsPerCapita(df):
	# replace & with and
	applyToCol(df, 'country', lambda x: re.sub(r'[\&]','and', str(x)))
	return df

def calcGunsPerCapita(df):
	df.GPC = df.guns / df.population * 100
	return df


def cleanPolicePerCapita(df):
	
	# get rid of (parent country name)
	applyToCol(df, 'country', lambda x: re.sub(r'[\(].*[\)]','', str(x)))
	
	# get rid of numbers
	applyToCol(df, 'country', lambda x: re.sub(r'[0-9]','', str(x)))
	
	# get rid of slashes
	applyToCol(df, 'country', lambda x: re.sub(r'[\/] ','', str(x)))

	# get rid of bracketed footnote references 
	applyToCol(df, 'PPC', lambda x: re.sub(r'[\[].*[\]]','', str(x)))

	return df


def rollupUK(df):
	topUK = 'United Kingdom'
	btmUK = ['Scotland','Northern Ireland','England and Wales']

	dfTopUK = df.loc[topUK, :]
	dfBtmUK = df.loc[btmUK, :]

	dictUK = dfTopUK.to_dict()

	
	
	dictUK.update({
		'country':topUK,
		'GPC':[np.average(dfBtmUK['GPC'], weights=dfBtmUK['population_x'])],
		'PPC':[np.average(dfBtmUK['PPC'], weights=dfBtmUK['population_x'])],
		'guns':[np.sum(dfBtmUK['guns'])],
		'population_x':[np.sum(dfBtmUK['population_x'])]
		})


	dfUK = pd.DataFrame(dictUK)
	dfUK = dfUK.set_index('country')

	df = df.drop(topUK)
	df = df.append(dfUK)

	return df

def supplementPolicePerCapita(df):

	copsExtra = {
		'Afghanistan':160000, # https://en.wikipedia.org/wiki/Afghan_National_Police#:~:text=The%20number%20of%20the%20Afghan,by%20the%20end%20of%202014.
		'Angola':math.nan,
		'Burkina Faso':math.nan,
		'Burundi':math.nan,
		'Central African Republic':1350, # https://en.wikipedia.org/wiki/Law_enforcement_in_the_Central_African_Republic#:~:text=The%20size%20of%20the%20National,police%20officers%20in%20the%20country.
		'DR Congo':125000, # https://en.wikipedia.org/wiki/Congolese_National_Police#:~:text=The%20Congolese%20National%20Police%20(French,answering%20to%20the%20Interior%20Ministry.
		'Egypt':500000, # https://en.wikipedia.org/wiki/Egyptian_National_Police
		'Guyana':4600, # https://www.stabroeknews.com/2020/01/31/news/guyana/police-force-has-to-be-expanded/
		'Honduras':13000, # https://www.insightcrime.org/news/brief/honduras-aims-to-double-national-police-by-2022/
		'Iraq':77000, # https://en.wikipedia.org/wiki/Law_enforcement_in_Iraq
		'Nicaragua':math.nan,
		'Rwanda':math.nan,
		'Sudan':math.nan,
		'Syria':28000, # https://en.wikipedia.org/wiki/Law_enforcement_in_Syria#cite_note-TSOAlW13Oct16-76
		'Venezuela':33000+8000+2400+50000, # https://en.wikipedia.org/wiki/Law_enforcement_in_Venezuela#:~:text=Law%20enforcement%20in%20Venezuela%20is,to%20the%20Ministry%20of%20Defence.
		}

	dfCops = pd.DataFrame.from_dict(copsExtra, orient = 'index', columns = ['cops'])

#	print(dfCops)

	df = merge(df, dfCops)	

	df['PPC2'] = df['cops'] / df['population_x'] * 100_000

	
	df.PPC[df.PPC2 > 0] = df.PPC2

	return df


def supplementKillingsPerCapita(df):
	pkpchistConfig = {
			'url': 'https://en.wikipedia.org/wiki/List_of_killings_by_law_enforcement_officers_by_country',
			'stats': {'PKPCHIST':None},
			'table': 1
		}

	pkhist = retrieveStats(pkpchistConfig)
	pkhist_avg = pkhist.mean(axis=1,skipna=True)
	pk2 = pd.DataFrame(pkhist_avg)
	pk2.columns = ['PKHIST']

	df2 = merge(df, pk2, how='inner')

	df2['PKPC2'] = df2.PKHIST / df2.population_x * 10_000_000

	df = merge(df, pd.DataFrame(df2.PKPC2))
	df.PKPC[df.PKPC2 > 0] = df.PKPC2

	return df

def getOffsetImage(path, zoom):
    return OffsetImage(plt.imread(path), zoom=zoom)

#class WikiStats:
#	def __init__(self, url, tableindex, countryCol, statCol, extraStatsNames=[], extraStatsCols=[], cleanupFunc=None):
#		self.url = url
#		self.tableIndex = tableIndex
#		self.countryCol = countryCol
#		self.statCol = statCol
#		self.extraStatsnames = extraStatsNames
#		self.extraStatsCols = extraStatsCols

####
#### from https://stackoverflow.com/questions/19073683/matplotlib-overlapping-annotations-text
####

def get_text_positions(text, x_data, y_data, txt_width, txt_height):
	a = list(zip(y_data, x_data))
	text_positions = list(y_data)
	print(a)
	print(text_positions)
	for index, (y, x) in enumerate(a):
		local_text_positions = [i for i in a if i[0] > (y - txt_height) 
							and (abs(i[1] - x) < txt_width * 2) and i != (y,x)]
		if local_text_positions:
			sorted_ltp = sorted(local_text_positions)
			print(sorted_ltp)
			if abs(sorted_ltp[0][0] - y) < txt_height: #True == collision
				differ = np.diff(sorted_ltp, axis=0)
				a[index] = (sorted_ltp[-1][0] + txt_height, a[index][1])
				text_positions[index] = sorted_ltp[-1][0] + txt_height*1.01
				for k, (j, m) in enumerate(differ):
					#j is the vertical distance between words
					if j > txt_height * 2: #if True then room to fit a word in
						a[index] = (sorted_ltp[k][0] + txt_height, a[index][1])
						text_positions[index] = sorted_ltp[k][0] + txt_height
						break
	print(text_positions)
	return text_positions

def text_plotter(text, x_data, y_data, text_positions, txt_width,txt_height):
	for z,x,y,t in zip(text, x_data, y_data, text_positions):
		print(z, x, y, t)
		plt.annotate(str(z), xy=(x-txt_width/2, t), size=12, zorder=50)
		if y != t:
			plt.arrow(x, t,0,y-t, color='red',alpha=0.3, width=txt_width*0.1, 
				head_width=txt_width, head_length=txt_height*0.5, 
				zorder=50,length_includes_head=True)





def buildStats():

	pd.set_option('chained_assignment',None)

	wikiConfig = [{
			'url': 'https://en.wikipedia.org/wiki/List_of_countries_by_income_equality',
			'stats': {'GINI':3},
			'table': 2,
			'drop': lambda x: x.dropna()
		},{
			'url': 'https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(PPP)_per_capita',
			'stats': {'PPP':2},
			'country': 1,
			'table':   4
		},{
			'url': 'https://en.wikipedia.org/wiki/Estimated_number_of_civilian_guns_per_capita_by_country',
			'stats': {'GPC':2,'population':5,'guns':6},
			'country': 1,
			'drop': lambda x: x.drop([0]),
			'clean': cleanGunsPerCapita,
			'calc':  calcGunsPerCapita
		},{
			'url': 'https://en.wikipedia.org/wiki/List_of_countries_and_dependencies_by_number_of_police_officers',
			'stats': {'PPC':3},
			'table': 2,
			'clean': cleanPolicePerCapita
		},{
			'url': 'https://en.wikipedia.org/wiki/List_of_killings_by_law_enforcement_officers_by_country',
			'stats': {'PKPC':8,'population':7}
		},{
			'url': 'https://en.wikipedia.org/wiki/Polity_data_series',
			'stats': {'POLIDX':3},
			'table': 1
		},{
			'url': 'https://en.wikipedia.org/wiki/Freedom_in_the_World',
			'stats': {'FWIDX':21}
		}]

	stats = {}
	for x in wikiConfig:
		stats.update({first(x['stats'].keys()):retrieveStats(x)})

	merged = mergemulti(list(stats.values()))

#	for s in stats:
#		colToFloat(merged, s)

	# combine scotland, england/wales, northern ireland
	merged = rollupUK(merged)

	merged = supplementPolicePerCapita(merged)

	merged = supplementKillingsPerCapita(merged)



	merged['GG'] = merged['PKPC'] * (merged['population_x']  / 10_000_000) / merged['guns'] * 1_000_000
	merged['GDP'] = merged['PPP'] * merged['population_x'] 

	return merged

def run_wls(df):

	wls = sm.WLS(df['PKPC'], df['GPC'], df['population_x'])

	res = wls.fit()

	print(res.params)

	#print(sm.stats.anova_lm(res))

	return

def get_non_overlap_coords(areas, textboxes, initpoints):

	return


def makeCharts(merged):




	reduce = merged[(merged['PKPC'] >= 0) & (merged['GDP'] > 100_000_000_000) & (merged['PPP'] > 25000)]

	reduce.GPC *= 10
	reduce.PKPC = np.maximum(reduce.PKPC, 0.1)

#	xlab = "$ GDP Per Capita, PPP"
#	ylab = "Number of Killings"
#	chartTitle = "Police Killings Per Million Civilian Firearms"

	xlab = "Civilian Firearms Per Thousand People"
	ylab = "Police Killings Per 10 Million People"
	chartTitle = "Police Killings Versus Number of Guns"

	def getFlagPath(country):
		c = country.lower().replace(' ','-')
		c = {
			'poland':'republic-of-poland',
			'united-states':'united-states-of-america'
			}.get(c, c)
			
		return os.path.join(FLAG_DIR, c + '.png')


	def run_matplotlib(df):

		ppp = df.PPP
		gpc = df.GPC
		pkpc = df.PKPC
		gg = df.GG
		sq = np.maximum(1,df.GPC ** 0.6)
	#	sq = df['population_x']

#		plt.scatter(ppp, gg, s=sq, alpha=0.4, edgecolors="grey", linewidth=2)
		plt.scatter(gpc, pkpc, s=sq, alpha=0.4, edgecolors="grey", linewidth=2)
		plt.grid(b=True, which='major', linestyle=':',color='#bbbbbb')
		plt.grid(b=True, which='minor', linestyle=':',color='#dddddd')

		ax = plt.gca()

		plt.suptitle(chartTitle, fontsize=28)
		ax.set_title('Developed countries. GDP per Capita > $25K. GDP > $100B. ($ PPP)', fontsize = 18)
		ax.set_xlabel(xlab, fontsize=16)
		ax.set_ylabel(ylab, fontsize=16)

		xlim = 1500

		ax.set_yscale('log')
		ax.set_xscale('log')
		ax.set_ylim([0.1,50])
		ax.set_xlim([0.1,xlim])

		from matplotlib.ticker import ScalarFormatter

		ax.yaxis.set_major_formatter(ScalarFormatter())
		ax.xaxis.set_major_formatter(ScalarFormatter())

		ax.set_yticks([0.5,1,5,10,50])
		ax.set_xticks([1,5,10,50,100,500,1000])

		ax.set_yticklabels(['0.5','1','5','10','50'], fontsize=16)
		ax.set_xticklabels(['1','5','10','50','100','500','1k'], fontsize=16)

		# Add titles (main and on axis)

		max_gpc = max(df.GPC)
		max_gg = max(df.GG)
		max_pkpc = max(df.PKPC)
		max_ppc = max(df.PPC)
		max_pop = max(df.population_x)

		fig = plt.gcf()
		dpi = float(fig.get_dpi())
		xsize = 1600
		ysize = xsize * 9/16
		fig.set_size_inches(xsize/dpi, ysize/dpi)

#		df = df.sort_values(by='PPC', ascending=False)

		cc = coco.CountryConverter()
		
		df = df.sort_values(by='population_x', ascending=False)
		gpc = df.GPC
		pkpc = df.PKPC

		for c in df.index:
#			iso3 = c
			iso3 = cc.convert(c, to='ISO3')

			pop = df.population_x[c]
			GPC = df.GPC[c]
			PKPC = df.PKPC[c]
#			PPC = df.PPC[c]
#			guns = df.GPC[c] / 100 * pop
#			PPP = df.PPP[c]
#			GG  = df.GG[c]
#			PPG = df.PPC[c] / 100_000 * pop / guns * 1000

			imgpath = getFlagPath(c)
#			imgzoom = max(0.05, gpc / max_gpc * 0.25)
			imgzoom = max(0.025, pop / max_pop * 0.2)

			img = plt.imread(imgpath)
			dims = len(img)
			# no more than 5% of horizontal space
			maxzoom = xsize / dims * 0.1

			imgzoom = (pop / max_pop) ** (1/2) * maxzoom




			point = (GPC, PKPC)
			ab = AnnotationBbox(OffsetImage(img, imgzoom), point, frameon=False)
			ax.add_artist(ab)

#			offsetbox = TextArea(iso3, minimumdescent=False)

#			ab = AnnotationBbox(offsetbox, point,
##                    xybox=(1.02, xy[1]),
##                    xycoords='data',
##                    boxcoords=("axes fraction", "data"),
#                    box_alignment=(5, 3),
#                    arrowprops=dict(arrowstyle="-"),
#					bboxprops=dict(edgecolor='#999999'),
#					pad = 0.2)
#			ax.add_artist(ab)

		x = gpc
		y = pkpc
		s = df.index
		texts = [plt.text(x[i], y[i], s[i], ha='center', va='center', zorder=50, fontsize=12.5)
			for i in range(len(x))]

		for t in texts:
			t.set_bbox(dict(edgecolor='#9999ff', facecolor='#9999ff', alpha=0.7, boxstyle='square,pad=0.05'))

		adjust_text(texts)


		plt.text(0.1, .06, "Source: Wikipedia", ha='left',va='bottom', fontsize=14, color='#666688')
		plt.text(xlim, .06, "@JDBurdick", ha='right',va='bottom', fontsize=14, color='#666688')

#		plt.annotate()

		#x = reduce.GPC
		#y = reduce.PKPC
		#text = reduce.index

		#txt_height = 0.0037*(plt.ylim()[1] - plt.ylim()[0])
		#txt_width = 0.018*(plt.xlim()[1] - plt.xlim()[0])

#		text_positions = get_text_positions(text, x, y, txt_width, txt_height)

#		text_plotter(text, x, y, text_positions, txt_width, txt_height)




#			plt.annotate(iso3, point)
#			plt.annotate(c + '\n' + 'Guns Per 100: %.1f' % gpc + '\n' + 'Police / 1k guns: %.0f' % ppg , (PPP, GG))

		plt.savefig('junk12.png')
		plt.show()

	def run_plotnine(df):

		sq = np.maximum(0,df['GPC'])
	#	sq = df['population_x']

		labs = p9.labels.labs(
					title = 'Police Killings Per Million Civilian Firearms',
					x = xlab,
					y = ylab,
					caption = '@JDBurdick')
		x = (
			ggplot(df)
				+ aes('PPP','GG')
				+ p9.scales.scale_y_continuous(trans='log10')
				+ geom_point(aes(size=sq))
				+ labs
		)

		x.draw()
		x.save('junk6.svg')

		return

	run_matplotlib(reduce)
#	run_plotnine(reduce)
##


dfStats = buildStats()

makeCharts(dfStats)


