from app import app

def test_hello_world():
  client = app.test_client()
  response = client.get('/')

  assert response.stutus_code == 200
  assert b'Hello World!' in response.data
