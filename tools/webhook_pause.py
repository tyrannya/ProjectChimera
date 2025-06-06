from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route('/telegram', methods=['POST'])
def telegram_webhook():
    data = request.get_json(force=True)
    if data.get('message', {}).get('text') == '/pause':
        requests.post('http://localhost:8080/api/v1/stop')
        return jsonify({'status': 'paused'})
    return jsonify({'status': 'ignored'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
