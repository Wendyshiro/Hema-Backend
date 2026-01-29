const express = require("express");
const router = express.Router();
const { testAPI } = require("../controllers/predictController");

router.get("/test", testAPI);

module.exports = router;
