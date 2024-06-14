# Breast Cancer Detection | Web App Demo (Flask) | convert to js (npm/node)
Breast Cancer Detection using Web App (Flask) that can classify if patient has Pneumonia or not based on uploaded MRI image.

Source dataset image :
(https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images/data)


## run this project with Flask (Python)
- **Follow these Steps**

  > Open the terminal/CMD in project directory

  > Then create virtual environment 

  > Install all the requirements using: 
      ```pip install -r requirements.txt```

  > If using NVDIA / CUDA, install it some drivers

  > dont forget to install tensorRT

  > After successful download of all above requirements, run the app using:
      ``` flask run ```
      Wait for few seconds till it shows like : ```Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)```
  > Then open this URL in browser : http://127.0.0.1:5000/


## run this project with NPM/JS
- **Follow these Steps**

  > Initialize a new Node.js project with `npm init -y`.

  > if available package.json, you can type `npm install`

  > (optional when not all library didnt install)Install necessary dependencies with `npm install express multer tensorflow tfjs-node ejs jimp`.

  > If didnt exist model_tfjs folder, you can type on terminal `tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model saved_model model_tfjs`

  > Start the server with `node app.js`.

  > Navigate to http://localhost:3000 in your browser to test the upload functionality.

  > folder and file js : index.html, app.js, model_tfjs
