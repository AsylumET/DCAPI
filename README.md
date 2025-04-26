# DCAPI
DCAPI - Demeter Core API

Overview
DCAPI is a lightweight, secure, and versatile RESTful API designed to serve as the backbone for data management in the Demeter greenhouse management software and other software built around Aegis. Likely standing for "Demeter Core API" (though its creator's sleep-coding sessions leave the "C" with a hint of mystery), DCAPI provides a dynamic interface for performing Create, Read, Update, and Delete (CRUD) operations on any configured database table.
Built with Flask, it supports both JSON and XML responses, fine-grained role-based access control (RBAC), and is optimized for deployment on resource-constrained Raspberry Pi devices in an intranet environment.
Whether handling sensor data for temperature control or managing plant inventory, DCAPI delivers a flexible, production-ready solution for Demeter’s data needs.

Table of Contents
    • Key Features
    • Architecture
    • Security Considerations
    • Deployment Considerations
    • Use Cases
    
Key Features
Dynamic CRUD Operations
    • Create: /create/<table> endpoint for single or bulk creation with validation and sanitization.
    • Read: /read/<table> and /read/<table>/<entry_id> with pagination, sorting, filtering.
    • Update: /update/<table> enables safe bulk updates.
    • Delete: /delete/<table> supports verified bulk deletion.
Batch Processing reduces network overhead, ideal for large datasets.
Role-Based Access Control (RBAC)
    • Fine-grained access associated with API keys and roles.
    • Defined through api_config.ini without code changes.
Security Features
    • API Key Authentication: Required via headers (or session-based for Swagger UI).
    • Session-Based Authentication: Secure sessions (1-hour lifetime).
    • Input Sanitization: Against SQL, XML, and other injections.
    • Password Hashing: Automatically hashes fields containing "password".
    • Rate Limiting: Via flask-limiter backed by Redis.
Flexible Response Formats
    • JSON (default) or XML responses.
    • Escaped XML output for legacy system compatibility.
Performance Optimization
    • Caching: Read caching via Redis or in-memory.
    • Connection Pooling: Minimal pool tuned for Raspberry Pi environments.
    • Table Metadata Caching: At startup to reduce reflection overhead.
Comprehensive Documentation
    • Swagger UI at /apidocs.
    • Web-based auth and docs access at /docs.
Health Monitoring
    • /health endpoint checks API and DB status.
Extensibility
    • Designed to support any table listed in ALLOWED_TABLES.
    • All keys, roles, limits, and DB settings configurable via api_config.ini.

Architecture
    • Framework: Flask
    • Database: SQLAlchemy (PostgreSQL or SQLite)
    • Caching: Flask-Caching with Redis
    • Rate Limiting: Flask-Limiter
    • Documentation: Flasgger (Swagger UI)
    • Security: Werkzeug (password hashing), middleware (API key validation)
    • Logging: Rotating file logs (1MB max, 5 backups)
Middleware intercepts non-static routes for API key validation and access logging.

Security Considerations
    • Authentication: API keys validated against api_config.ini.
    • Authorization: Fine-grained role and column-level access.
    • Input Validation: Type, length, and required field checks.
    • Data Protection: Password fields hashed.
    • Session Security: Short-lived sessions with environment-protected keys.
    • Rate Limiting: Redis-backed quotas to prevent abuse.

Deployment Considerations
    • Target: Raspberry Pi or similar low-power devices.
    • Offline Installation: All dependencies bundled as .whl files.
    • Database Options: PostgreSQL (preferred) or SQLite (fallback).
    • Server: Single-worker Gunicorn instance.
    • Redis: Local instance for caching and rate limiting.
    • Logging: Rotated logs to prevent storage overflow.
    • Monitoring: /health endpoint; /stats endpoint planned.
Deployment Steps (example for Raspberry Pi):
    1. Bundle dependencies and copy to Pi.
    2. Install dependencies:
       pip install --no-index --find-links=/path/to/wheels -r requirements.txt
    3. Configure api_config.ini and environment variables.
    4. Run Gunicorn:
       gunicorn -w 1 -b 0.0.0.0:5000 your_app:app
    5. Access via intranet clients (cURL, Swagger UI).

Use Cases
    • IoT Greenhouse Hub: Log sensor data from Pi-based devices.
    • Internal Portal: Manage plant inventory, staff schedules, and sensor configurations.
    • Offline Analytics: Collect and process greenhouse data without internet access.
Example:
    • Log temperature and humidity sensor data to a sensors table.
    • Admin users update configurations.
    • Read-only users monitor sensor outputs.
    • Bulk operations streamline large data uploads.

License
GNU Affero General Public License v3.0

Demeter ✨ - Empowering Greenhouse Innovation.
