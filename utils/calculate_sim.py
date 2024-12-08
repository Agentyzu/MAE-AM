import requests
import json

# Constants for API credentials
API_KEY = ""       # API_KEY
SECRET_KEY = ""    # SECRET_KEY


def sim(text1, text2):
    """
    Calculates the similarity score between two texts using the Baidu NLP API.

    Args:
        text1 (str): The first text to compare.
        text2 (str): The second text to compare.

    Returns:
        float: The similarity score between the two texts.
    """
    url = "https://aip.baidubce.com/rpc/2.0/nlp/v2/simnet?charset=UTF-8&access_token=" + get_access_token()
    payload = json.dumps({
        "text_1": text1,
        "text_2": text2
    })
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    # print("text1:", text1)
    # print("text2:", text2)
    # print(response.text)
    return json.loads(response.text)["score"]


def get_access_token():
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))
