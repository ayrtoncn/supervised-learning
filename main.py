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

app = Flask(__name__)
api = Api(app)

class TestDataInterpreter():
	def __init__(self, filename = "", data_stat = []):
		self.filename = filename
		self.data = []
		self.data_stat = data_stat
		self.processInputFile()

	def processInputFile(self):
		input_data = self.getInputFileContent()
		self.makeDatasetList(input_data)
		
		for i in range(len(self.data)):
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
			self.data.append(row[0].split(','))

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
        inp = pickle.load(open('inp.sav', 'rb'))
        clf = pickle.load(open('clf.sav', 'rb'))

    def get(self):
        data = request.args.get('data')
        try:
            len(data)
        except:
            return
        test_data = TestDataInterpreter(filename="tubes2_HeartDisease_test.csv", data_stat=self.inp.data_stat)
        result = self.clf.predict(test_data)
        return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)