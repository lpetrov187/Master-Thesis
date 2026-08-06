# requests: Sessions and Connection Pooling

The `requests` library provides a `Session` object that persists parameters
across requests and reuses the underlying TCP connection. Creating a new
`requests.get(...)` call each time opens a new connection; using a
`Session` instead lets `requests` pool and reuse connections via
`urllib3`'s `HTTPAdapter`.

```python
import requests

session = requests.Session()
response = session.get("https://api.example.com/users")
```

By default, a `Session` uses an `HTTPAdapter` with a connection pool sized
`pool_connections=10` and `pool_maxsize=10`. For applications issuing many
concurrent requests to the same host, mount a custom adapter with a larger
pool:

```python
adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20)
session.mount("https://", adapter)
```

Reusing a `Session` across requests reduces latency from repeated TCP and
TLS handshakes and is the recommended pattern for any code that makes more
than a handful of requests to the same host.
