from flask import Flask, render_template, request, redirect, url_for, session,send_file
from werkzeug.utils import secure_filename
import os
import Model_ABSA as model

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'csv','xls'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    return render_template('index.html')
    
@app.route('/home')
def homepage():
    return render_template("snippets.html")
    
@app.route('/Analytics')
def chart():
    return render_template("chart.html")

@app.route('/download/<filename>')
def download_file(filename):
    # Validate that the file exists (you should add more error handling here)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        return "File not found", 404
    # Use the provided filename for the attachment
    return send_file(file_path, as_attachment=True)



@app.route('/upload', methods=['POST'])
def upload():   
    if request.method == 'POST':
        review = request.form.get("message",None)
        file = request.files.get('file',None)    
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        if review or file:
            if not file:
                filename=None
            else:
                filename=os.path.join(app.config['UPLOAD_FOLDER'], filename)
            result=model.main(review,filename)
            table=result["review"]
            filename=result["file"]
            if table=="":
                table="""<div class="row d-flex justify-content-center">-</div>"""
            if filename != "":
                content="""<div class="row d-flex justify-content-center">
                                 <p class="custom-font">Click beneath the download button to retrieve the reviews.</p></div>
                        <div class="row justify-content-center">
                        <button type="button" class="btn btn-warning" id="openUrlButton">Download</button>
                        <script>
                                document.getElementById("openUrlButton").addEventListener("click", function() {"""+"""
                                        // Define the URL you want to open in a new tab
                                        var urlToOpen = '/download/{}'; // Replace with your desired URL
                                        
                                        // Open the URL in a new tab
                                        window.open(urlToOpen, '_blank');""".format(filename)+"""
                                });
                                                     
                        </script> </div>"""
            else:
                content="""<div class="row d-flex justify-content-center">-</div>"""
            return render_template('result.html',res=table,downloadcontent=content)

if __name__ == '__main__':
    app.run(debug=False,host="0.0.0.0")
