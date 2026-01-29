// index.js
require("dotenv").config();
const express = require("express");
const cors = require("cors");

const predictRoutes = require("./src/routes/predictRoutes");

const app = express();

// Middlewares
app.use(cors());
app.use(express.json()); // allows JSON request bodies

// Routes
app.use("/api/predict", predictRoutes);

// Port
const PORT = process.env.PORT || 5000;

app.listen(PORT, () => {
  console.log(`HEMA backend running on port ${PORT}`);
});
