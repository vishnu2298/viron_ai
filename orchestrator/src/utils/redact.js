/** Strip common PII (US + India formats) before anything is persisted or used as training data. */
const digits = (s) => s.replace(/\D/g, '').length;

const RULES = [
  [/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/gi, '[EMAIL]'],
  [/\b(?:sk|pk|api|key)[-_][A-Za-z0-9_-]{16,}\b/g, '[SECRET]'],
  [/\b\d{3}-\d{2}-\d{4}\b/g, '[SSN]'],
  [/\b[A-Z]{5}\d{4}[A-Z]\b/g, '[PAN]'],
  [/\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{1,7}\b/g, (m) => (digits(m) >= 13 ? '[CARD]' : m)],
  [/\b\d{4}[ -]?\d{4}[ -]?\d{4}\b/g, '[AADHAAR]'],
  // Phone: optional +CC, then digit groups; only redact when 10-13 digits total
  [/(?:\+\d{1,3}[\s-]?)?\(?\d{2,5}\)?(?:[\s-]?\d{2,5}){1,3}/g, (m) => (digits(m) >= 10 && digits(m) <= 13 ? '[PHONE]' : m)],
];

function redact(text) {
  if (typeof text !== 'string') return text;
  return RULES.reduce((s, [re, rep]) => s.replace(re, rep), text);
}
module.exports = { redact };
