# pytest: Fixtures

A fixture is a function decorated with `@pytest.fixture` that pytest runs
before a test that requests it, typically to set up state and yield it.

```python
import pytest

@pytest.fixture
def sample_data():
    return {"id": 1, "name": "example"}

def test_uses_fixture(sample_data):
    assert sample_data["id"] == 1
```

Fixtures default to function scope, meaning pytest calls the fixture again
for every test function that requests it. Passing `scope="module"` or
`scope="session"` to `@pytest.fixture` reuses the same fixture value across
all tests in a module or the whole test session, which is useful for
expensive setup like opening a database connection.

Fixtures can also depend on other fixtures by naming them as arguments, and
pytest resolves the dependency graph automatically.
