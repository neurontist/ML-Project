from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.pipeline.train_pipeline import Training

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        gender = request.form.get('gender')
        race_ethnicity=request.form.get('ethnicity')
        parental_level_of_education=request.form.get('parental_level_of_education')
        lunch=request.form.get('lunch')
        test_preparation_course=request.form.get('test_preparation_course')
        reading_score=float(request.form.get('writing_score'))
        writing_score=float(request.form.get('reading_score'))


        data = CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score
        )

        features = data.format_data()

        train = Training()
        success_training = train.fit()

        if success_training == "yes": 
            model = PredictPipeline()

            preds = model.predict(features=features)
            print("Successful")
            return render_template('home.html', success="Successful", results=preds[0])
        else:
            return render_template('home.html', success="Not Successful", results="Error")

    
if __name__ == '__main__':
    app.run(host='0.0.0.0')
    