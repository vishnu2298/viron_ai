const Joi = require('joi');
const chatSchema = Joi.object({
  message: Joi.string().trim().min(1).max(8000).required(),
  history: Joi.array()
    .items(Joi.object({ role: Joi.string().valid('user', 'assistant').required(), content: Joi.string().max(8000).required() }))
    .max(20)
    .default([]),
  options: Joi.object({
    tier: Joi.string().valid('tiny', 'core'),
    allowEscalation: Joi.boolean(),
  }).default({}),
});
const validate = (schema) => (req, res, next) => {
  const { error, value } = schema.validate(req.body, { abortEarly: false, stripUnknown: true });
  if (error) {
    return res
      .status(400)
      .json({ success: false, message: 'Validation failed', code: 'VALIDATION_ERROR', details: error.details.map((d) => d.message) });
  }
  req.body = value;
  next();
};
module.exports = { validate, chatSchema };
