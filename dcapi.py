from flasgger import Swagger
from flask import send_from_directory
from flask import Flask, render_template, url_for, request, jsonify, Response, abort, redirect, session
from flask import make_response
from flask_caching import Cache
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash
from flask_babel import LazyString
from sqlalchemy import MetaData, Table, select, insert, update, delete, asc, desc, text, Boolean, Integer, String
from sqlalchemy.exc import IntegrityError
from xml.etree.ElementTree import Element, SubElement
import xml.etree.ElementTree as ET
from logging.handlers import RotatingFileHandler
import logging
import configparser
import time
import json
import os
import re

# ──────── Access Logging ────────

# Setup logger
api_logger = logging.getLogger("api_access")
api_logger.setLevel(logging.INFO)
handler = RotatingFileHandler(
      os.path.join(os.path.dirname(__file__), 'api_access.log'),
      maxBytes=1000000,
      backupCount=5
)
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)
api_logger.addHandler(handler)

app = Flask(
    __name__, 
    template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
    static_folder='static'
)

# ──────── Generate OpenAPI Documentation ────────

app.config['SWAGGER'] = {
    'uiversion': 3,
    'swagger_ui_template': 'swaggerui.html',
    'doc_expansion': 'list'  # options: 'none', 'list', 'full'
}

swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": LazyString(lambda: "DCAPI"),
        "version": LazyString(lambda: "1.0"),
        "description": LazyString(lambda: "A lightweight, secure, and versatile RESTful API designed to serve as the backbone for data management in the Demeter greenhouse management software and other software built around Aegis. Likely standing for \"Demeter Core API\" (though its creator's sleep-coding sessions leave the \"C\" with a hint of mystery), DCAPI provides a dynamic interface for performing Create, Read, Update, and Delete (CRUD) operations on any configured database table.\n\nBuilt with Flask, it supports both JSON and XML responses, fine-grained role-based access control (RBAC), and is optimized for deployment on resource-constrained Raspberry Pi devices in an intranet environment.\n\nWhether handling sensor data for temperature control or managing plant inventory, DCAPI delivers a flexible, production-ready solution for Demeter’s data needs."),
    },
}

swagger_config = {
    'headers': [],
    'specs': [
        {
            'endpoint': 'apispec_1',
            'route': '/apispec_1.json',
            'rule_filter': lambda rule: True,  # all endpoints
            'model_filter': lambda tag: True,  # all models
        }
    ],
    'static_url_path': '/flasgger_static',
    'swagger_ui': True,
    'specs_route': '/apidocs/'
}

swagger = Swagger(app, template=swagger_template, config=swagger_config)

# ──────── Flask Caching ────────

ENV = os.getenv('APP_ENV', 'development')

if ENV == 'production':
    cache_config = {
        'CACHE_TYPE': 'RedisCache',
        'CACHE_REDIS_HOST': 'localhost',
        'CACHE_REDIS_PORT': 6379,
        'CACHE_DEFAULT_TIMEOUT': 300
    }
else:
    cache_config = {
        'CACHE_TYPE': 'SimpleCache',
        'CACHE_DEFAULT_TIMEOUT': 300
    }

app.config.from_mapping(cache_config)
cache = Cache(app)
cache.init_app(app)

# ──────── API Keys & Roles ────────

config = configparser.ConfigParser()
try:
    config.read(os.path.join(os.path.dirname(__file__), 'api_config.ini'))
    if not config.sections():
        raise configparser.Error("Config file is empty or missing")
except configparser.Error as e:
    print(f"Failed to load config: {e}")
    exit(1)

app.secret_key = os.getenv('SECRET_KEY', config.get('APP', 'secret_key', fallback=os.urandom(24).hex()))

API_KEYS = dict(config["API_KEYS"])
ALLOWED_TABLES = dict(config["ALLOWED_TABLES"])
ROLE_COLUMN_ACCESS = { # Role-Based Access Control (RBAC)
    role: json.loads(access)
    for role, access in config["ROLE_COLUMN_ACCESS"].items()
}

db_user, db_pass, db_host, db_name = (config["DB"][k] for k in ('user', 'pass', 'host', 'name'))

default_limits = {
    "storage_uri": "redis://localhost:6379",
    "create": "1 per second",
    "read": "10 per second",
    "update": "10 per second",
    "delete": "1 per second"
}
for key, value in default_limits.items():
    config["RATELIMITING"].setdefault(key, value)
rl_create, rl_read, rl_update, rl_delete = (config["RATELIMITING"][k] for k in ('create', 'read', 'update', 'delete'))

# ──────── Configuration ────────
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour

# If PostgreSQL is too resource-intensive, switch to SQLite
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL", f"postgresql://{db_user}:{db_pass}@{db_host}/{db_name}")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = { # Configure SQLAlchemy connection pooling
      'pool_size': 2,
      'max_overflow': 2,
      'pool_timeout': 30,
}

# ──────── Rate Limiting ────────

rate_limit_backend = config.get('RATELIMITING', 'storage_uri', fallback='memory://')
limiter = Limiter(get_remote_address, app=app, default_limits=["100 per minute"], storage_uri=rate_limit_backend)

# ──────── Database Setup ────────
db = SQLAlchemy(app)
metadata = MetaData()

# ──────── Error Handling ────────

def error_dict(status, message):
    return {
        "status": status, 
        "message": str(message)
    }

@app.errorhandler(400)
def bad_request(e):
    return jsonify(error_dict('error', e.description)), 400

@app.errorhandler(403)
def forbidden(e):
    return jsonify(error_dict('error', e.description)), 403

@app.errorhandler(404)
def not_found(e):
    return jsonify(error_dict('error', e.description)), 404

@app.errorhandler(429)
def too_many_requests(e):
    return jsonify(error_dict('error', e.description)), 429

@app.errorhandler(500)
def internal(e):
    return jsonify(error_dict('error', e.description)), 500

# ──────── Helpers ────────

def get_api_key():
    """Returns API key from request."""    
    return session.get('api_key') or request.headers.get("X-API-Key")

def get_api_role():
    """Returns role associated with API key."""
    api_key = get_api_key()
    return API_KEYS.get(api_key)

def get_allowed_columns(table):
    """Returns allowed columns for the table based on API role."""
    role = get_api_role()
    return ROLE_COLUMN_ACCESS.get(role, {}).get(table, [])

def rows_to_xml(rows, table_name, column_format=None):
    """Converts row dicts to XML with optional format change."""
    root = ET.Element(table_name)
    
    # If 'short' format is requested, use different XML structure
    if column_format:
        for col in rows:
            ET.SubElement(root, 'column').text = escape_xml(col)

    else:
        for row in rows:
            if isinstance(row, dict):
                for col, val in row.items():
                    if isinstance(val, list):
                        sub_element = ET.SubElement(root, col)
                        for item in val:
                            ET.SubElement(sub_element, 'column').text = escape_xml(item)
                    else:
                        ET.SubElement(root, col).text = escape_xml(val)
            elif isinstance(row, (list, tuple)):
                for val in row:
                    ET.SubElement(root, singularize(table_name[:-1])).text = escape_xml(val)
            else:
                ET.SubElement(root, singularize(table_name[:-1])).text = escape_xml(row)
    
    return ET.tostring(root, encoding='utf8')

def escape_xml(val):
    """Proper XML escaping."""
    if val is None:
        return ''
    return str(val).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&apos;')

def singularize(word):
    """Converts simple English plurals to singular form."""
    if word.endswith('ies'):
        return word[:-3] + 'y' # categories -> category
    elif word.endswith('ses') or word.endswith('xes') or word.endswith('zes'):
        return word[:-2] # fixes -> fix, boxes -> box
    elif word.endswith('s') and not word.endswith('ss'):
        return word[:-1] # tables -> table, but not "glass" -> "glas"
    else:
        return word

def format_response(data, output_format, table_name, column_format=None):
    """Formats response as JSON or XML."""
    if output_format == 'xml':
        if isinstance(data, dict):
            data = [data]  # Wrap dicts
        elif not isinstance(data, list):
            data = [[data]]  # Wrap single values
        xml_data = rows_to_xml(data, table_name, column_format)
        return Response(xml_data, mimetype='application/xml')
    return jsonify(data)

def sanitize_input(data, max_length=255):
    """Sanitize input data to prevent common issues."""
    sanitized = {}
    for key, value in data.items():
        if not isinstance(value, (str, int, float, bool, type(None))):
            raise ValueError(f"Invalid type for {key}: {type(value)}")
        if isinstance(value, str):
            if len(value) > max_length:
                api_logger.warning(f"Truncated {key} from {len(value)} to {max_length} characters")
            value = value.strip()[:max_length]
            value = value.replace('&', '&').replace('<', '<').replace('>', '>').replace('"', '"').replace("'", '&apos;')
        sanitized[key] = value
    return sanitized

TABLE_CACHE = {} # Cache table objects at startup
def load_table(table_name):
    """Loads a SQLAlchemy Table object if it's allowed."""
    if table_name not in TABLE_CACHE:
        if table_name not in ALLOWED_TABLES:
            abort(403, description="Forbidden: Access to this table is not allowed.")
        TABLE_CACHE[table_name] = Table(ALLOWED_TABLES[table_name], metadata, autoload_with=db.engine, extend_existing=True)

    return TABLE_CACHE[table_name]

INSPECTOR_CACHE = {}
def get_inspector():
    """Cache the inspector results."""
    if 'inspector' not in INSPECTOR_CACHE:
        INSPECTOR_CACHE['inspector'] = db.inspect(db.engine)

    return INSPECTOR_CACHE['inspector']

def validate_tables():
    """Validate table schemas at startup to catch configuration errors early."""
    with app.app_context():
        with db.engine.begin() as conn:
            for table_name, actual_name in ALLOWED_TABLES.items():
                try:
                    Table(actual_name, metadata, autoload_with=db.engine)
                except Exception as e:
                    print(f"Invalid table {table_name}: {e}")
                    exit(1)

# Call after app and db are configured
validate_tables()

def make_cache_key():
    return request.full_path

def clear_table_cache(table, *args):
    """
    Clears the cache for a given read_all, read_one and table. 
    Optionally supports additional args like entry_id.
    """
    for fmt in ['json', 'xml']:
        cache.delete_memoized(read_all, table, fmt)
        cache.delete_memoized(read_one, table, fmt)
        if args:
            cache.delete_memoized(read_all, table, fmt, *args)
            cache.delete_memoized(read_one, table, fmt, *args)

def is_valid_password_hash(value: str) -> bool:
    if not isinstance(value, str):
        return False

    # Werkzeug formats look like: method$salt$hash
    parts = value.split('$')
    if len(parts) != 3:
        return False

    method, salt, hashval = parts

    # Accept only known/expected hashing methods
    allowed_methods = {'pbkdf2:sha256', 'pbkdf2:sha512', 'scrypt'}
    if method not in allowed_methods:
        return False

    # Salt and hash should be reasonably long and alphanumeric/base64-like
    if not re.fullmatch(r'[A-Za-z0-9./]{8,}', salt):
        return False
    if not re.fullmatch(r'[A-Za-z0-9./]{20,}', hashval):
        return False

    return True

# ──────── Middleware ────────

@app.before_request
def check_api_key():
    """Middleware to check API key for all non-static requests."""
    allowed_paths = ['/apidocs/', '/apidocs', '/swagger', '/swagger.json', '/apispec_1.json']
    if (
        request.path.startswith('/flasgger_static')
        or request.path.startswith('/static')
        or any(request.path.startswith(p) for p in allowed_paths)
    ):
        # Only try to get the API key if it's one of the allowed paths
        if any(request.path.startswith(p) for p in allowed_paths):
            api_key = get_api_key()
            if api_key not in API_KEYS:
                abort(403, description="Forbidden: API key required for documentation access")
        return

    # For all other routes, require API key and log access
    api_key = get_api_key()
    role = get_api_role()

    # Log access
    api_logger.info(f"API key: {api_key} | Role: {role} | Path: {request.path} | Method: {request.method} | IP: {request.remote_addr}")

# ──────── Routes ────────

@app.route('/tables', defaults={'output_format': 'json'}, methods=['GET'])
@app.route('/tables/<output_format>', methods=['GET'])
@limiter.limit(rl_read)
@cache.memoize(timeout=300)
def list_tables(output_format):
    """
    List all table names from the database.
    ---
    parameters:
      - name: output_format
        in: path
        type: string
        required: false
    responses:
      200:
          description: List of tables retrieved successfully
    """
    try:
        table_names = [t for t in ALLOWED_TABLES.keys()]
        return format_response(table_names, output_format, table_name="tables")
    except DatabaseError as e:
        api_logger.error(f"Database error during table list: {e}")
        abort(500, description="Database connection error")
    except Exception as e:
        api_logger.error(f"Unexpected error during table list: {e}")
        abort(500, description="Internal server error")

@app.route('/columns/<table>', defaults={'output_format': 'json'}, methods=['GET'])
@app.route('/columns/<table>/<output_format>', methods=['GET'])
@limiter.limit(rl_read)
@cache.memoize(timeout=300)
def list_columns(table, output_format):
    """
    List columns for a specific table.
    ---
    parameters:
      - name: table
        in: path
        type: string
        required: true
      - name: output_format
        in: path
        type: string
        required: false
    responses:
      200:
          description: List of columns retrieved successfully
      404:
          description: Table not found
    """
    try:
        tbl = load_table(table)  # Enforces ALLOWED_TABLES
        columns = sorted([col.name for col in tbl.columns])
        return format_response(columns, output_format, table_name=f"columns_of_{table}", column_format=True)
    except DatabaseError as e:
        api_logger.error(f"Database error during column list: {e}")
        abort(500, description="Database connection error")
    except Exception as e:
        api_logger.error(f"Unexpected error during column list: {e}")
        abort(500, description="Internal server error")

@app.route('/tables_with_columns', defaults={'output_format': 'json'}, methods=['GET'])
@app.route('/tables_with_columns/<output_format>', methods=['GET'])
@limiter.limit(rl_read)
@cache.memoize(timeout=300)
def list_tables_with_columns(output_format):
    """
    List all tables with their columns or list columns from multiple specific tables at once.
    ---
    parameters:
      - name: tables
        in: query
        type: string
        required: false
        description: Comma-separated list of table names (optional).
      - name: output_format
        in: path
        type: string
        required: false
        description: Desired output format (json, xml, etc.)
    responses:
      200:
        description: Tables and columns retrieved successfully
        content:
          application/json:
            schema:
              type: object
              additionalProperties:
                type: array
                items:
                  type: string
              example:
                users: ["id", "name", "email"]
                sensors: ["id", "timestamp", "value"]
      400:
        description: Invalid table names
      500:
        description: Database or internal error
    """
    if ENV != 'production':
        cache.delete_memoized(list_tables_with_columns)

    try:
        inspector = get_inspector()
        tables_param = request.args.get('tables', None)
        available_tables = inspector.get_table_names()

        if tables_param:

            tables = [t.strip() for t in tables_param.split(',') if t.strip()]
            if not tables or any(not re.match(r'^[a-zA-Z0-9_]+$', t) for t in tables):
                abort(400, description="Invalid table names")

            tables = [t for t in tables if t in ALLOWED_TABLES]
            tbl_name = "batch_columns"
        else:
            tables = list(ALLOWED_TABLES.keys())
            tbl_name = "tables_with_columns"

        result = {}
        for table in tables:
            if table not in available_tables:
                result[table] = 'Table not found'
            else:
                tbl = load_table(table)
                allowed_cols = get_allowed_columns(table)
                columns = sorted(col.name for col in tbl.columns if col.name in allowed_cols)
                result[table] = columns
        return format_response(result, output_format, table_name=tbl_name)

    except DatabaseError as e:
        api_logger.error(f"Database error during tables_with_columns: {e}")
        abort(500, description="Database connection error")
    except Exception as e:
        api_logger.error(f"Unexpected error during tables_with_columns: {e}")
        abort(500, description="Internal server error")

@app.route('/create/<table>', methods=['POST'])
@limiter.limit(rl_create)
def create_entries(table):
    """
    Create one or more new entries in a given table.
    ---
    parameters:
      - name: table
        in: path
        type: string
        required: true
        description: The name of the table to insert into
      - name: data
        in: body
        required: true
        schema:
          type: object
          oneOf:
            - description: Single entry
              properties:
                column_name:
                  type: string
            - description: Multiple entries
              type: array
              items:
                type: object
                properties:
                  column_name:
                    type: string
    responses:
      200:
        description: Entry or entries created successfully
        schema:
          type: object
          properties:
            ids:
              type: array
              items:
                type: integer
            status:
              type: string
      400:
        description: Invalid input, missing required fields, or constraint violation
      403:
        description: Forbidden access
      500:
        description: Internal server error
    """
    if get_api_role() == "readonly":
        abort(403, description="Forbidden access")
    data = request.get_json()
    if not data:
        abort(400, description="Missing JSON data")
    entries = data if isinstance(data, list) else [data]
    tbl = load_table(table)
    allowed_cols = get_allowed_columns(table)
    inserted_ids = []
    # Pre-validation to ensure all entries use only allowed columns
    disallowed_entries = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            abort(400, description="Each entry must be a JSON object")
        invalid_keys = [k for k in entry if k not in tbl.c or k not in allowed_cols]
        if invalid_keys:
            disallowed_entries.append({"index": i, "invalid_columns": invalid_keys})
    if disallowed_entries:
        abort(400, description=f"Disallowed column usage in entries: {disallowed_entries}")
    for entry in entries:
        entry = {k: v for k, v in entry.items() if v not in [None, ""]}
        try:
            entry = sanitize_input(entry)
        except ValueError as e:
            abort(400, description=str(e))
        invalid_keys = [k for k in entry if k not in tbl.c.keys()]
        if invalid_keys:
            abort(400, description=f"Invalid column(s): {', '.join(invalid_keys)}")
        values = {k: v for k, v in entry.items()
                  if k in allowed_cols and k in tbl.c.keys() and k != 'id'}
        missing_required = [
            col.name for col in tbl.columns
            if not col.nullable and not col.default and not col.server_default and col.name not in values
        ]
        if missing_required:
            abort(400, description=f"Missing required field(s): {', '.join(missing_required)}")
        for col_name, column in tbl.c.items():
            if col_name in values:
                val = values[col_name]
                if isinstance(column.type, Boolean):
                    if val is None:
                        abort(400, description=f"Boolean value for {col_name} cannot be null")
                    if isinstance(val, str):
                        values[col_name] = val.strip().lower() in ['true', '1', 'yes']
                    elif isinstance(val, (bool, int)):
                        values[col_name] = bool(val)
                    else:
                        abort(400, description=f"Invalid boolean value for {col_name}")
                elif isinstance(column.type, Integer):
                    try:
                        values[col_name] = int(val)
                    except (ValueError, TypeError):
                        abort(400, description=f"Invalid value for {col_name}, expected integer")
                elif isinstance(column.type, String):
                    val_str = str(val)
                    if hasattr(column.type, 'length') and column.type.length:
                        if len(val_str) > column.type.length:
                            abort(400, description=f"{col_name} exceeds max length of {column.type.length}")
                    values[col_name] = val_str
                    if 'password' in col_name.lower() and not is_valid_password_hash(val):
                        values[col_name] = generate_password_hash(str(val))
        stmt = insert(tbl).values(**values).returning(tbl.c.id)
        try:
            with db.engine.begin() as conn:
                inserted_id = conn.execute(stmt).scalar()
            inserted_ids.append(inserted_id)
        except IntegrityError as e:
            api_logger.error(f"Constraint violation during create: {e}")
            abort(400, description="Duplicate entry or constraint violation")
        except DatabaseError as e:
            api_logger.error(f"Database error during create: {e}")
            abort(500, description="Database connection error")
        except Exception as e:
            api_logger.error(f"Unexpected error during create: {e}")
            abort(500, description="Internal server error")
    clear_table_cache(table)
    return jsonify({"ids": inserted_ids, "status": "created"})

@app.route('/read/<table>/<int:entry_id>', defaults={'output_format': 'json'}, methods=['GET', 'POST'])
@app.route('/read/<table>/<int:entry_id>/<output_format>', methods=['GET', 'POST'])
@limiter.limit(rl_read)
@cache.memoize(timeout=300)
def read_one(table, entry_id, output_format):
    """
    Read a specific entry by ID from a table.
    ---
    parameters:
      - name: table
        in: path
        type: string
        required: true
      - name: entry_id
        in: path
        type: integer
        required: true
      - name: output_format
        in: path
        type: string
        required: false
    responses:
      200:
          description: Entry retrieved successfully
      404:
          description: Entry not found
    """
    tbl = load_table(table)
    stmt = select(tbl).where(tbl.c.id == entry_id)
    try:
        with db.engine.begin() as conn:
            row = conn.execute(stmt).mappings().first()
            if not row:
                api_logger.error(f"Error during read: Not Found")
                abort(404, description="Not found: The entry cannot be found.")
            allowed_cols = get_allowed_columns(table)
            row_filtered = {k: v for k, v in dict(row).items() if k in allowed_cols}
            return format_response(row_filtered, output_format, table)            
    except DatabaseError as e:
        api_logger.error(f"Database error during read: {e}")
        abort(500, description="Database connection error")
    except Exception as e:
        api_logger.error(f"Unexpected error during read: {e}")
        abort(500, description="Internal server error")

@app.route('/read/<table>', defaults={'output_format': 'json'}, methods=['GET', 'POST'])
@app.route('/read/<table>/<output_format>', methods=['GET', 'POST'])
@limiter.limit(rl_read)
@cache.memoize(timeout=300)
def read_all(table, output_format):
    """
    Read all entries in a table with optional pagination and sorting.
    ---
    parameters:
      - name: table
        in: path
        type: string
        required: true
      - name: output_format
        in: path
        type: string
        required: false
    """
    start = int(request.args.get("start", 0))
    limit = min(int(request.args.get("limit", 100)), 1000) # Cap pagination limits at 1000
    sort_by = request.args.get("sort_by")
    order = request.args.get("order", "asc")
    tbl = load_table(table)
    stmt = select(tbl)
    if sort_by and sort_by in tbl.c:
        stmt = stmt.order_by(asc(tbl.c[sort_by]) if order == "asc" else desc(tbl.c[sort_by]))
    for key, value in request.args.items():
        if key in tbl.c and key not in ['start', 'limit', 'sort_by', 'order']:
            stmt = stmt.where(tbl.c[key] == value)
    stmt = stmt.offset(start).limit(limit)
    try:
        with db.engine.begin() as conn:
            rows = conn.execute(stmt).mappings().all()
            if not rows:
                api_logger.error(f"Error during read: Not Found")
                abort(404, description="Not found: The entries cannot be found.")
            allowed_cols = get_allowed_columns(table)
            rows_filtered = [
                {k: v for k, v in dict(row).items() if k in allowed_cols}
                for row in rows
            ]
            return format_response(rows_filtered, output_format, table)
    except DatabaseError as e:
        api_logger.error(f"Database error during read: {e}")
        abort(500, description="Database connection error")
    except Exception as e:
        api_logger.error(f"Unexpected error during read: {e}")
        abort(500, description="Internal server error")

@app.route('/update/<table>', methods=['POST'])
@limiter.limit(rl_update)
def update_entries(table):
    """
    Update one or more entries in a table by ID.
    ---
    parameters:
      - name: table
        in: path
        type: string
        required: true
        description: Name of the table to update.
      - name: updates
        in: body
        required: true
        schema:
          type: array
          items:
            type: object
            properties:
              id:
                type: integer
                description: ID of the entry to update.
              column_name:
                type: string
                description: Value to update for the given column.
    responses:
      200:
        description: Entries updated successfully.
        schema:
          type: object
          properties:
            updated_ids:
              type: array
              items:
                type: integer
              description: List of updated entry IDs.
      400:
        description: Invalid columns or input data.
      403:
        description: Forbidden Read-only access.
      500:
        description: Internal server error.
    """
    if get_api_role() == "readonly":
        abort(403, description="Forbidden: Read-only access.")
    data = request.get_json(force=True)
    updates = [data] if isinstance(data, dict) else data if isinstance(data, list) else None
    if not updates:
        abort(400, description="Invalid update payload")
    tbl = load_table(table)
    allowed_cols = get_allowed_columns(table)
    updated_ids = []
    try:
        with db.engine.begin() as conn:
            # for entry in updates:
            disallowed_updates = []
            for entry in updates:
                entry_id = entry.get("id")
                if entry_id is None:
                    abort(400, description="Each update must include an 'id' field")
                # Check for disallowed fields before processing
                disallowed = [k for k in entry if k != 'id' and (k not in allowed_cols or k not in tbl.c)]
                if disallowed:
                    disallowed_updates.append({"id": entry_id, "disallowed": disallowed})
            if disallowed_updates:
                msg_lines = [f"ID {x['id']}: {', '.join(x['disallowed'])}" for x in disallowed_updates]
                abort(400, description="Disallowed column(s) for update:\n" + "\n".join(msg_lines))
            # If all clear, continue with actual updates
            for entry in updates:
                entry_id = entry.get("id")
                # Validate and sanitize fields
                values = {}
                for k, v in entry.items():
                    if k == 'id' or k not in allowed_cols or k not in tbl.c:
                        continue
                    try:
                        sanitized = sanitize_input({k: v})
                        v = sanitized[k]
                    except ValueError as e:
                        abort(400, description=str(e))
                    column = tbl.c[k]
                    if isinstance(column.type, Boolean):
                        if v is None:
                            abort(400, description=f"Boolean value for {k} cannot be null")
                        if isinstance(v, str):
                            values[k] = v.strip().lower() in ['true', '1', 'yes']
                        elif isinstance(v, (bool, int)):
                            values[k] = bool(v)
                        else:
                            abort(400, description=f"Invalid boolean value for {k}")
                    elif isinstance(column.type, Integer):
                        try:
                            values[k] = int(v)
                        except (ValueError, TypeError):
                            abort(400, description=f"Invalid value for {k}, expected integer")
                    elif isinstance(column.type, String):
                        val_str = str(v)
                        if hasattr(column.type, 'length') and column.type.length:
                            if len(val_str) > column.type.length:
                                abort(400, description=f"{k} exceeds max length of {column.type.length}")
                        values[k] = val_str
                        if 'password' in k.lower() and not is_valid_password_hash(v):
                            values[k] = generate_password_hash(str(v))
                if not values:
                    continue
                stmt = update(tbl).where(tbl.c.id == entry_id).values(**values)
                conn.execute(stmt)
                updated_ids.append(entry_id)
                clear_table_cache(table, entry_id)
        return jsonify({"updated_ids": updated_ids, "status": "updated"})
    except Exception as e:
        api_logger.error(f"Error during bulk update: {e}")
        abort(500, description=f"Internal Error: {str(e)}")

@app.route('/delete/<table>', methods=['POST'])
@limiter.limit(rl_delete)
def delete_entries(table):
    """
    Delete one or more entries by ID from a table.
    ---
    parameters:
      - name: table
        in: path
        type: string
        required: true
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            properties:
              entry_ids:
                oneOf:
                  - type: array
                    items:
                      type: integer
                  - type: integer
    responses:
      200:
        description: Entries deleted successfully
      400:
        description: Invalid input
      403:
        description: Forbidden Read-only access
      500:
        description: Internal Error
    """
    if get_api_role() == "readonly":
        abort(403, description="Forbidden: Read-only access.")
    entry_ids = request.json.get("entry_ids")
    if isinstance(entry_ids, int):
        entry_ids = [entry_ids]
        if not all(isinstance(eid, int) for eid in entry_ids):
           abort(400, description="'entry_ids' must only contain integers.")
    elif not isinstance(entry_ids, list) or not entry_ids:
        abort(400, description="'entry_ids' must be an integer or non-empty list.")
    tbl = load_table(table)
    try:
        with db.engine.begin() as conn:
            # Validate entry_ids existence
            existing_ids = conn.execute(select(tbl.c.id).where(tbl.c.id.in_(entry_ids))).scalars().all()
            missing_ids = set(entry_ids) - set(existing_ids)
            if missing_ids:
                abort(404, description=f"Entries not found: {', '.join(map(str, missing_ids))}")
            stmt = delete(tbl).where(tbl.c.id.in_(entry_ids))
            conn.execute(stmt)
        for entry_id in entry_ids:
            clear_table_cache(table, entry_id)
        return jsonify({"deleted_ids": entry_ids})
    except Exception as e:
        api_logger.error(f"Error during delete: {e}")
        abort(500, description=f"Internal Error: {str(e)}")

@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint.
    ---
    responses:
      200:
        description: Health check successful
      500:
        description: Health check failed
    """
    start = time.perf_counter()
    try:
        with db.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_latency = (time.perf_counter() - start) * 1000  # milliseconds
        return jsonify({"status": "ok", "db_latency_ms": round(db_latency, 2)})
    except Exception as e:
        api_logger.error(f"Health check failed: {e}")
        abort(500, description="Health check failed.")

# TODO: Uncomment when instaled on RaspberryPi
# @app.route('/stats', methods=['GET'])
# def stats():
    # import psutil
    # return jsonify({
        # "cpu_percent": psutil.cpu_percent(),
        # "memory_percent": psutil.virtual_memory().percent
    # })

@app.route('/docs', methods=['GET', 'POST']) 
def docs():
    """
    An endpoint to authenticate for web-based API access.
    ---
    responses:
      200:
          description: Authenticated successfully
      403:
          description: Forbidden Read-only access
      500:
          description: Internal Error
    """
    success=None
    error=None
    api_key=None
    if request.method == 'POST':
        # Form was submitted
        api_key = request.form.get('api_key')
        if not api_key:
            api_logger.error("Error: API key is required.")
            # Handle missing API key
            return render_template('docs.html', success=None, error="API key is required.", api_key=None)
        else:
            # Validate the API key
            if api_key not in API_KEYS:
                return render_template('docs.html', success=None, error="Invalid API key.", api_key=None)
            api_logger.info("Success: API key was valid.")
            # Store API key in session
            session['api_key'] = api_key
            # Handle valid API key - return redirect('/apidocs/')
            return render_template('docs.html', success=True, error=None, api_key=api_key)
    api_logger.info("No POST request, rendering default.")
    # For initial GET request or first visit
    return render_template('docs.html', success=None, error=None, api_key=None)

# ──────── Run Server ────────
if __name__ == '__main__':
    app.run(debug=True)
