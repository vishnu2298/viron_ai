class AppError extends Error {
  constructor(message, statusCode = 500, code = 'APP_ERROR', details) {
    super(message);
    this.statusCode = statusCode;
    this.code = code;
    this.details = details;
  }
}
class UpstreamError extends AppError {
  constructor(service, message) {
    super(`${service}: ${message}`, 502, 'UPSTREAM_ERROR');
  }
}
module.exports = { AppError, UpstreamError };
