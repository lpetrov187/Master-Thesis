# Python logging: Basic Configuration

The standard library's `logging` module is configured once, usually near
program startup, via `logging.basicConfig` or by building handlers
manually.

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)
logger.info("service started")
```

Each module should call `logging.getLogger(__name__)` rather than using the
root logger directly, so log lines are attributable to the module that
produced them and so per-module log levels can be tuned independently.

`basicConfig` only has an effect the first time it's called; calling it
again in the same process is a no-op unless `force=True` is passed. This
trips people up when a library call sets up logging before the
application's own `basicConfig` call runs.
