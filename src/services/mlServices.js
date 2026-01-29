const axios = require("axios");
const PYTHON_API = process.env.PYTHON_API || "http://localhost:5000";

exports.sendData = async (symptoms) => {
  try {
    const response = await axios.post(`${PYTHON_API}/risk/analyze`, symptoms, {
      headers: { "Content-Type": "application/json" },
      timeout: 5000,
    });
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.message || error.message || "Python service error");
  }
};
