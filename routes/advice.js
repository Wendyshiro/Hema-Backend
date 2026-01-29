const express = require('express');
const router = express.Router();
const Advice = require('../models/Advice');

// GET /advice?category=Prevention
router.get('/', async (req, res) => {
  const category = req.query.category;

  if (!category) {
    return res.status(400).json({ message: 'Category is required' });
  }

  try {
    // Case-insensitive search using a regular expression
    const adviceList = await Advice.find({
      Category: new RegExp(`^${category}$`, 'i')
    });

    res.json(adviceList);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Server error' });
  }
});

module.exports = router;
