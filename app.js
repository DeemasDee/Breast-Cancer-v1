const express = require('express');
const multer = require('multer');
const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const fs = require('fs');

const app = express();
const UPLOAD_FOLDER = 'static/uploads';
const MODEL_PATH = path.join(__dirname, 'model_tfjs/model.json');

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, UPLOAD_FOLDER);
  },
  filename: (req, file, cb) => {
    cb(null, file.originalname);
  }
});
const upload = multer({ storage });

app.use('/static', express.static(path.join(__dirname, 'static')));

// Load the trained model
let model;
async function loadModel() {
  model = await tf.loadGraphModel(`file://${MODEL_PATH}`);
}
loadModel();

// Function to preprocess the image for prediction
async function preprocessImage(imagePath) {
  const imageBuffer = fs.readFileSync(imagePath);
  const imageTensor = tf.node.decodeImage(imageBuffer, 3);
  const resizedImage = tf.image.resizeBilinear(imageTensor, [50, 50]);
  const expandedImage = resizedImage.expandDims();
  return expandedImage.div(255.0);
}


// Route to handle prediction request
app.post('/predict', upload.single('image'), async (req, res) => {
  const imgPath = path.join(UPLOAD_FOLDER, req.file.filename);
  const processedImg = await preprocessImage(imgPath);
  const prediction = model.predict(processedImg);
  const predictedClass = prediction.argMax(-1).dataSync()[0];
  const classNames = ['No Breast Cancer', 'Breast Cancer'];
  const result = classNames[predictedClass];
  const imageUrl = `/static/uploads/${req.file.filename}`;
  res.json({ result: result, image: imageUrl });
  //res.json({ result: result, image: imgPath });
});

// Route to render the prediction HTML page
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

app.listen(3000, () => {
  console.log('Server is running on http://localhost:3000');
});
