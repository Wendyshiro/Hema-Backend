const { spawn } = require('child_process'); 
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');

const adviceRoutes = require('./routes/advice');

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());

// Routes
app.use('/api/advice', adviceRoutes);

// POST endpoint for ML risk prediction
app.post('/api/predict-risk', (req, res) => {
  const userAnswers = req.body;

  console.log('📝 Flutter sent:', userAnswers); // 🔹 Log exactly what Flutter sent

  const pyProcess = spawn(process.env.PYTHON || 'python3', ['predict_risk.py']);

  let pythonOutput = '';
  let pythonError = '';
  let responded = false; // 🔹 Track if we've already sent a response

  pyProcess.stdout.on('data', (data) => {
    pythonOutput += data.toString();
  });

  pyProcess.stderr.on('data', (data) => {
    pythonError += data.toString();
    console.error('Python stderr:', data.toString());
  });

  // 🔹 Kill Python if it hangs (timeout)
  const timeout = setTimeout(() => {
    if (!responded) {
      responded = true;
      pyProcess.kill();
      console.error('⚠️ Python timeout');
      res.status(504).json({ error: 'Python process timeout' });
    }
  }, 15000); // 15s < Flutter 15s timeout

  pyProcess.on('close', (code) => {
    clearTimeout(timeout);

    if (responded) return; // 🔹 Prevent double responses
    responded = true;

    console.log('Raw Python output:', pythonOutput);

    if (code !== 0) {
      return res.status(500).json({
        error: 'Python script failed',
        details: pythonError || 'Unknown error'
      });
    }

    if (!pythonOutput) {
      return res.status(500).json({ error: 'No output from Python' });
    }

    try {
      const result = JSON.parse(pythonOutput.trim());
      console.log('✅ Prediction result:', result); // 🔹 Log what will be sent
      return res.json(result);
    } catch (err) {
      return res.status(500).json({
        error: 'Invalid JSON from Python',
        raw: pythonOutput
      });
    }
  });

  // 🔹 Send user answers to Python
  pyProcess.stdin.write(JSON.stringify(userAnswers));
  pyProcess.stdin.end();
});

// Start server WITHOUT waiting for MongoDB
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

// MongoDB connection
const mongoUri = 'mongodb://localhost:27017/hema';

mongoose
  .connect(mongoUri)
  .then(() => console.log('MongoDB connected'))
  .catch((err) => console.error('MongoDB connection error:', err));
