const mongoose = require('mongoose'); 
const AdviceSchema = new mongoose.Schema({
  Category: { type: String, required: true },
  Subcategory: { type: String, required: true },
  Topic: { type: String, required: true },
  Information: { type: String, required: true },
  Source_Type: { type: String },
  Target_Audience: { type: String },
});

module.exports = mongoose.model('Advice', AdviceSchema , 'hema');
