  # First get auth token
  TOKEN=$(curl -s -X POST "http://localhost:8000/auth/login" \
    -H "Content-Type: application/json" \
    -d '{"username":"brandon","password":"YOUR_PASSWORD"}' | \
    python3 -c "import sys, json; print(json.load(sys.stdin)['access_token'])")

  # Then query with profile
  curl -X POST "http://localhost:8000/chat" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"query":"YOUR_QUESTION_HERE","profile_name":"Workshop Facilitator"}' | \
    python3 -c "import sys, json; data=json.load(sys.stdin); print(data['response'])"

  Or as a true single command:

  curl -s -X POST "http://localhost:8000/auth/login" \
    -H "Content-Type: application/json" \
    -d '{"username":"brandon","password":"YOUR_PASSWORD"}' | \
  python3 -c "
  import sys, json, requests
  token = json.load(sys.stdin)['access_token']
  response = requests.post('http://localhost:8000/chat',
    headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'},
    json={'query': 'YOUR_QUESTION_HERE', 'profile_name': 'Workshop Facilitator'})
  print(response.json()['response'])"
