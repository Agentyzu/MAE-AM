import math

import requests

from utils.api_config import require_env, request_timeout


def sim(text1, text2):
    """Return the similarity score from the configured Baidu NLP-compatible service."""
    url, = require_env('SIMILARITY_URL')
    token = get_access_token()
    response = requests.post(url, params={'access_token': token},
                             json={'text_1': text1, 'text_2': text2},
                             timeout=request_timeout())
    response.raise_for_status()
    body = response.json()
    if 'score' not in body:
        raise ValueError('Similarity service response is missing score')
    score = float(body['score'])
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('Similarity service score must be between 0 and 1')
    return score


def get_access_token():
    """Request an access token without logging credentials or token values."""
    url, api_key, secret_key = require_env('SIMILARITY_AUTH_URL', 'SIMILARITY_API_KEY', 'SIMILARITY_SECRET_KEY')
    response = requests.post(url, params={'grant_type': 'client_credentials',
                                         'client_id': api_key, 'client_secret': secret_key},
                             timeout=request_timeout())
    response.raise_for_status()
    token = response.json().get('access_token')
    if not token:
        raise ValueError('Authentication service response is missing access_token')
    return token
