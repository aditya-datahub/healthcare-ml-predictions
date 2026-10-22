B=https://raw.githubusercontent.com/aditya-datahub/healthcare-ml-predictions/main
for f in heart-disease-prediction/README.md heart-disease-prediction/main.ipynb heart-disease-prediction/heart_disease_data.csv heart-disease-prediction/requirements.txt diabetes-prediction/README.md diabetes-prediction/main.ipynb diabetes-prediction/diabetes.csv diabetes-prediction/requirements.txt breast-cancer-prediction/README.md breast-cancer-prediction/main.ipynb breast-cancer-prediction/data.csv breast-cancer-prediction/requirements.txt LICENSE; do echo "$f: $(curl -sS -o /dev/null -w '%{http_code} %{size_download}' $B/$f)"; done
echo; curl -sS $B/diabetes-prediction/main.ipynb | python3 -I -c "
import json,sys
n=json.load(sys.stdin)
for c in n['cells']:
    s=''.join(c['source'])
    if 'std_data' in s or 'scaler.transform' in s: print(s[:300]); print('====')"
Response
heart-disease-prediction/README.md: 200 5710
heart-disease-prediction/main.ipynb: 200 49294
heart-disease-prediction/heart_disease_data.csv: 200 11328
heart-disease-prediction/requirements.txt: 200 34
diabetes-prediction/README.md: 200 6725
diabetes-prediction/main.ipynb: 200 47227
diabetes-prediction/diabetes.csv: 200 23873
diabetes-prediction/requirements.txt: 200 45
breast-cancer-prediction/README.md: 200 6732
breast-cancer-prediction/main.ipynb: 200 92622
breast-cancer-prediction/data.csv: 200 125204
breast-cancer-prediction/requirements.txt: 200 45
LICENSE: 200 1070

input_data = (10,168,74,0,0,38,0.537,34)

# changing the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)
print("Numpy Array: ",input_data_as_numpy_array)

# reshape the function as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape
====
