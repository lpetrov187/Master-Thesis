# SQLAlchemy: Configuring Connection Pooling

SQLAlchemy's `Engine` manages a connection pool by default, so application
code should create one `Engine` per database and reuse it rather than
creating a new engine per request.

```python
from sqlalchemy import create_engine

engine = create_engine(
    "postgresql://user:pass@localhost/dbname",
    pool_size=10,
    max_overflow=5,
    pool_timeout=30,
    pool_recycle=1800,
)
```

- `pool_size` sets the number of connections kept open in the pool.
- `max_overflow` allows temporary extra connections beyond `pool_size`
  under load.
- `pool_recycle` closes and replaces connections older than the given
  number of seconds, which avoids errors from database servers that close
  idle connections after a timeout.

For short-lived scripts, `NullPool` disables pooling entirely so every
connection is opened and closed immediately, which avoids leaving idle
connections around after the script exits.
