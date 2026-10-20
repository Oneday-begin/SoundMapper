from fastapi.testclient import TestClient
from soundmap.app import app


def test_welcome_and_workspace_are_separate_and_assets_resolve():
    with TestClient(app) as client:
        welcome = client.get('/')
        workspace = client.get('/app')
        assert welcome.status_code == workspace.status_code == 200
        assert 'href="/app"' in welcome.text
        assert 'href="/app?view=library"' in welcome.text
        assert '/static/app.js' not in welcome.text
        assert '/static/app.js' in workspace.text
        assert 'id="track-a"' in workspace.text
        assert client.get('/static/landing.css').status_code == 200
