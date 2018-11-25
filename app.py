import csv
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import preprocessing
import pickle
import warnings
from flask import jsonify
from flask_restful import Resource, Api
from flask import Flask, request
import os
app = Flask(__name__)
api = Api(app)

class InputDataInterpreter():
	def __init__(self, filename = ""):
		self.filename = filename
		self.data = []
		self.target = []
		self.data_stat = []
		self.processInputFile()
		self.initDataStatistic()

	def initDataStatistic(self):
		for j in range(len(self.data[0])):
			stat = {}
			
			for i in range(len(self.data)):
				stat['modus'] = self.getDataModus(j)
				stat['mean'] = self.getDataMean(j)
				stat['median'] = self.getDataMedian(j)
			
			self.data_stat.append(stat)
	
	def reduceUnknownData(self):
		pass

	def countUnknownAttr(self, row):
		count_unknown = 0
		for data in row:
			if data == '?':
				count_unknown += 1
		
		return count_unknown

	def processInputFile(self):
		input_data = self.getInputFileContent()

		self.makeDatasetList(input_data)
		
		for i in range(len(self.data)):
			self.target[i] = int(self.target[i])
			for j in range(len(self.data[0])):
				self.data[i][j] = float(self.data[i][j])	

	def getInputFileContent(self):
		data_content = []
		
		with open(self.filename, newline='') as csvfile:
			file_content = csv.reader(csvfile, delimiter=' ', quotechar='|')
			
			for row in file_content:
				content_row = []
				for data in row:
					content_row.append(data)
				data_content.append(content_row)

		return data_content[1:]

	def makeDatasetList(self, input_data):
		for row in input_data:
			self.target.append(row[0].split(',')[-1])
			self.data.append(row[0].split(',')[0:13])
		
		self.reduceUnknownData()
		
		self.patchUnknownData()

	def patchUnknownData(self):
		column_patch_method = ["median", "modus", "modus", "mean", \
		"mean", "modus", "modus", "mean", \
		"modus", "mean", "modus", "modus", "modus"]
		
		column_patch_values = self.getColumnPatchVal(column_patch_method)

		for i in range(len(self.data)):
			for j in range(len(self.data[0])):
				if self.data[i][j] == '?' or self.data[i][j] == '':
					self.data[i][j] = column_patch_values[j]


	def getColumnPatchVal(self, patch_method):
		patch_val = []

		for i in range(len(patch_method)):
			if patch_method[i] == 'modus':
				patch_val.append(self.getDataModus(i))
			elif patch_method[i] == 'median':
				patch_val.append(self.getDataMedian(i))
			elif patch_method[i] == 'mean':
				patch_val.append(self.getDataMean(i))

		return patch_val

	def getDataModus(self, j):
		data_dict = {}

		for i in range(len(self.data)):
			if self.data[i][j] == '?':
				continue
			if str(self.data[i][j]) in data_dict:
				data_dict[str(self.data[i][j])] += 1
			else :
				data_dict[str(self.data[i][j])] = 0

		max_key = ''
		max_val = -1
		for key, val in data_dict.items():
			if val > max_val:
				max_key = key
				max_val = val

		return str(max_val)

	def getDataMedian(self, j):
		column_list = []

		for i in range(len(self.data)):
			if self.data[i][j] == '?':
				continue
			column_list.append(self.data[i][j])

		column_list.sort()
		median_idx = (len(column_list)//2) + 1

		return str(column_list[median_idx])

	def __is_int__(self, input):
		try:
			a = int(input)
			return True
		except :
			return False

	def getDataMean(self, j):
		column_sum = 0
		data_num = 0

		for i in range(len(self.data[0])):
			if self.__is_int__(self.data[i][j]):
				column_sum += int(self.data[i][j])
				data_num += 1

		return str(column_sum / data_num)

class TestDataInterpreter():
	def __init__(self,data, data_stat = []):
		self.data = []
		self.data_stat = data_stat
		self.processInputFile(data)

	def processInputFile(self,data):
		self.makeDatasetList(data)
		for i in range(len(self.data)):
			for j in range(len(self.data[0])):
				self.data[i][j] = float(self.data[i][j])
		print(self.data)
	def makeDatasetList(self, input_data):
		self.data.append(input_data.split(','))

		self.patchUnknownData()

	def patchUnknownData(self):
		column_patch_method = ["median", "modus", "modus", "mean", \
		"mean", "modus", "modus", "mean", \
		"modus", "mean", "modus", "modus", "modus"]

		column_patch_values = self.getColumnPatchVal(column_patch_method)

		for i in range(len(self.data)):
			for j in range(len(self.data[0])):
				if self.data[i][j] == '?':
					self.data[i][j] = column_patch_values[j]


	def getColumnPatchVal(self, patch_method):
		patch_val = []

		for i in range(len(patch_method)):
			if patch_method[i] == 'modus':
				patch_val.append(self.data_stat[i]['modus'])
			elif patch_method[i] == 'median':
				patch_val.append(self.data_stat[i]['median'])
			elif patch_method[i] == 'mean':
				patch_val.append(self.data_stat[i]['mean'])

		return patch_val

class Checker(Resource):
    def __init__(self):
        self.inp = pickle.load(open('inp.sav', 'rb'))
        self.clf = pickle.load(open('clf.sav', 'rb'))

    def get(self):
        data = request.args.get('data')
        try:
            len(data)
        except:
            return
        test_data = TestDataInterpreter(data=data, data_stat=self.inp.data_stat)
		
        pred = self.clf.predict(test_data.data)
        result =    {
                    'Result' : int(pred[0])
                    }
        print(result)
        return jsonify(result)

api.add_resource(Checker, '/checker/')

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='127.0.0.1', port=port)